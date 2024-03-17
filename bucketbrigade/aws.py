from io import BytesIO
import base64
import csv
import ftplib
import io
import json
import mimetypes
from datetime import date, datetime, timezone
from pathlib import Path
from typing import List
from urllib.parse import quote_plus, unquote_plus, urlparse
from xml.etree import ElementTree as ET

import arrow
import boto3
import dateutil
import numpy as np
import pandas as pd
import pysftp
import s3fs
from botocore.exceptions import ClientError
from magika import Magika
from pydantic import BaseModel
from textractor import Textractor
from textractor.data.constants import TextractFeatures

from bucketbrigade import core as bbcore


s3 = s3fs.S3FileSystem(anon=False, default_cache_type="none")


def bucket_key_from_docpath(docpath):
    """
    Extracts the bucket name and key (prefix) from a full S3 document path.

    :param docpath: Full S3 path (e.g., 's3://bucket-name/prefix')
    :return: Tuple of (bucket_name, key)
    """
    full_path = docpath.split("//")[-1]
    bucket_name = full_path.split("/")[0]
    key = "/".join(full_path.split("/")[1:])
    return bucket_name, key


def url_encode_s3_path(s3_path: str) -> str:
    # Remove the s3:// prefix as it's not part of the path we want to encode
    path_without_scheme = s3_path.split("//")[-1]

    # Encode the path
    encoded_path = quote_plus(path_without_scheme)

    # Return the encoded path
    return encoded_path


def url_decode_s3_path(encoded_url: str) -> str:
    # Decode the URL-encoded string
    decoded_url = unquote_plus(encoded_url)

    # Return the decoded URL
    full_s3_path = f"s3://{decoded_url}"
    return full_s3_path


def list_docs(parent_folder, start=None, end=None, include_string=""):
    """
    Lists documents in an S3 bucket that are within the specified date range.

    :param docpath: Full S3 path to the bucket and optional prefix
    :param start: Start date as a string (optional)
    :param end: End date as a string (optional)
    :return: List of file paths in the S3 bucket that meet the criteria
    """
    try:
        s3 = boto3.client("s3")
        bucket_name, prefix = bucket_key_from_docpath(parent_folder)
        kwargs = {"Bucket": bucket_name, "MaxKeys": 1000}
        if prefix:
            kwargs["Prefix"] = prefix

        # Convert string dates to datetime objects, if provided
        if start:
            start = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
        if end:
            end = datetime.fromisoformat(end).replace(tzinfo=timezone.utc)

        files_list = []
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(**kwargs):
            for content in page.get("Contents", []):
                last_modified = content.get("LastModified")
                if (start and last_modified < start) or (
                    end
                    and last_modified > end
                    or (include_string and include_string not in content.get("Key"))
                ):
                    continue
                if content.get("Key")[-1] != "/":  # Skip directories/folders
                    files_list.append(f"s3://{bucket_name}/{content.get('Key')}")

        return files_list
    except ClientError as e:
        # Handle AWS client errors (e.g., authentication issues, access denied)
        print(f"An error occurred: {e}")
        return []
    except Exception as e:
        # Handle other exceptions (e.g., parsing date strings)
        print(f"An unexpected error occurred: {e}")
        return []


def get_output_docnames(
    parent_folder, include_string="", output_folders=["skip_folder", "output_folder"]
):
    """
    Generates a list of document names in specified output folders.

    Args:
    - metadata: An object containing various pieces of information, including the input path.
    - include_string: A string used for filtering documents in the output folders.

    Returns:
    - A list of document names present in the 'skip_folder' and 'output_folder'.
    """
    output_docpaths = []
    input_folder = parent_folder.split("/")[-1]
    for folder_name in output_folders:
        # Construct the path for the current folder
        output_path = parent_folder.replace(f"/{input_folder}", f"/{folder_name}")
        # List documents in the current folder and extract their names
        output_docpaths.extend(
            [
                x.split("/")[-1]
                for x in list_docs(output_path, include_string=include_string)
            ]
        )
    return output_docpaths


def get_secrets(secret_name, profile_name="mf", suffix="", account="422890657323"):
    region_name = "ap-southeast-2"
    if suffix:
        arn = f"arn:aws:secretsmanager:{region_name}:{account}:secret:{secret_name}{suffix}"
    else:
        arn = f"arn:aws:secretsmanager:{region_name}:{account}:secret:{secret_name}"
    try:
        session = boto3.session.Session()
    except:
        session = boto3.session.Session(profile_name=profile_name)
    client = session.client(service_name="secretsmanager", region_name=region_name)
    return json.loads(client.get_secret_value(SecretId=arn)["SecretString"])


def get_docpaths_to_process(parent_folder, include_string=""):
    """
    Identifies documents in the input path that need to be processed.

    Args:
    - metadata: An object containing various pieces of information, including the input path.
    - include_string: A string used for filtering documents in the input path.

    Returns:
    - A list of document paths from the input folder that are not present in the output folders.
    """
    # List documents in the input path
    input_docpaths = list_docs(parent_folder, include_string=include_string)
    if include_string:
        input_docpaths.extend(
            [
                x
                for x in list_docs(
                    parent_folder.replace("/input_folder", "/archive_folder"),
                    include_string=include_string,
                )
            ]
        )
        if not input_docpaths:
            input_docpaths.extend(
                [
                    x
                    for x in list_docs(
                        parent_folder.replace("/input_folder", "/skip_folder"),
                        include_string=include_string,
                    )
                ]
            )
    # Get names of documents in the output folders
    # output_docnames = get_output_docnames(parent_folder, include_string=include_string)
    output_docnames = []
    # Filter out documents that are already in the output folders
    return [x for x in input_docpaths if x.split("/")[-1] not in output_docnames]


def make_dated_filename(docpath, timezone="Australia/Sydney"):
    short_timezone_today = arrow.utcnow().to(timezone).format("YYYYMMDDHHmmss")
    docpath_parts = docpath.rsplit("/", 1)
    if len(docpath_parts) == 2:
        docpath_parts[0] = f"{docpath_parts[0]}/"
    docname = docpath_parts[-1]
    try:
        date_found = dateutil.parser.parse(docname[:8], yearfirst=True)
        return f"{docpath}"
    except:
        return f"{docpath_parts[0]}{short_timezone_today}_{docname}"


def create_content(docpath):
    bucket_name, prefix = bucket_key_from_docpath(docpath)
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=bucket_name, Key=prefix)
    body = obj["Body"]
    content = body.read()
    try:
        content = content.decode("utf-8")
    except:
        pass
    return body, content


def doc_exists(docpath, s3_additional_kwargs=None, boto3_session=None, version_id=None):
    """Check if a file exists in S3."""
    s3 = boto3.client("s3")
    bucket, key = bucket_key_from_docpath(docpath)

    try:
        if version_id:
            response = s3.head_object(Bucket=bucket, Key=key, VersionId=version_id)
        else:
            response = s3.head_object(Bucket=bucket, Key=key)

        return True
    except ClientError as ex:
        if ex.response["Error"]["Code"] == "404":
            return False
        raise ex


def copy_doc(src_path, destination_parent_path, new_docname=None):
    # Get the docname from the source path
    docname = src_path.split("/")[-1]

    # Check if new_docname is provided, use it instead of the original docname
    if new_docname:
        docname = new_docname

    # Handle case where destination_parent_path is the same as src_path
    if src_path.split("/")[-1] == destination_parent_path.split("/")[-1]:
        destination_parent_path = destination_parent_path.replace(
            src_path.split("/")[-1], ""
        )

    # Ensure the destination_parent_path ends with a '/'
    if not destination_parent_path.endswith("/"):
        destination_parent_path += "/"

    # Get the bucket and key from the source path
    src_bucket, src_key = bucket_key_from_docpath(src_path)

    # Append the docname (or new_docname) to the destination_parent_path
    dest_path = destination_parent_path + docname
    print(dest_path)

    # Get the bucket and key from the destination path
    dest_bucket, dest_key = bucket_key_from_docpath(dest_path)

    # Initialize the boto3 S3 client
    s3 = boto3.client("s3")

    # Copy the file from the source bucket to the destination bucket
    s3.copy_object(
        CopySource={"Bucket": src_bucket, "Key": src_key},
        Bucket=dest_bucket,
        Key=dest_key,
    )

    return dest_path


def skip_doc(current_path, delete_original=True):
    if doc_exists(current_path):
        new_path = current_path.replace("input_folder", "skip_folder")
        stripped_path = "/".join(new_path.strip("/").split("/")[:-1]) + "/"
        copy_doc(current_path, stripped_path)
        if delete_original and doc_exists(new_path):
            delete_doc(current_path)
        print(f"Moved doc to {new_path}")
    return ""


def read_doc(docpath, sheet_name=0, cat=False):
    body, content = create_content(docpath)

    if cat:
        return content

    if isinstance(content, str):
        try:
            json_data = bbcore.parse_xml(content, remove_line_breaks=False)
            return json.dumps(json_data)
        except:
            pass

        try:
            return json.loads(content)
        except:
            pass

        try:
            num_commas = [line.count(",") for line in content.splitlines()]
            if len(num_commas) > 0 and all(count > 0 for count in num_commas):
                try:
                    csv.reader(content, delimiter=",")
                    return pd.read_csv(docpath)
                except:
                    pass
        except:
            pass

        return content

    try:
        df = pd.read_excel(docpath, sheet_name=sheet_name)
        print("Excel")
        return df
    except:
        pass

    try:
        df = pd.read_parquet(docpath, engine="fastparquet")
        if "index" in df.columns:
            df = df.set_index(["index"])
        return df
    except:
        pass

    return content


def convert_df_list_column_to_json(df):
    if df.shape[0]:
        for col in df.columns:
            if isinstance(df.iloc[0][col], bytes):
                try:
                    df[col] = df[col].apply(lambda x: x.decode("utf-8"))
                except:
                    pass
            if isinstance(df.iloc[0][col], list):
                df[col] = df[col].apply(lambda x: json.dumps(x))
                df[col] = df[col].astype(str)
    return df


def save_doc(docpath, doc, pipe=False, dated=True, timezone="Australia/Sydney"):
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, (date, datetime)):
                return obj.isoformat()
            elif isinstance(obj, np.floating):
                return float(obj)

    if dated:
        docpath = make_dated_filename(docpath, timezone)
    s3 = boto3.client("s3")
    bucket_name, prefix = bucket_key_from_docpath(docpath)

    if pipe:
        try:
            doc = base64.b64decode(doc)
        except Exception:
            try:
                doc = doc.encode("utf-8")
            except Exception as e:
                print("Not base64 encoded or utf-8 encoded")
                print(e)
                pass

        s3.put_object(Body=doc, Bucket=bucket_name, Key=prefix)
        return {"bytes": docpath}
    try:
        doctype = "ET"
        doc = ET.tostring(doc, encoding="unicode")
    except:
        pass

    doctype = "text"
    if isinstance(doc, pd.Series):
        doc = pd.DataFrame([doc.to_dict()])
        doctype = "dataframe"
    if isinstance(doc, pd.DataFrame):
        doctype = "dataframe"
        if doc.index.name not in doc.columns:
            try:
                doc = doc.reset_index()
            except:
                pass
        doc = convert_df_list_column_to_json(doc)
        doc = doc.convert_dtypes()
    elif isinstance(doc, ET.Element):
        doctype = "ET"
        doc = bbcore.parse_xml(ET.tostring(doc, encoding="unicode"))
    elif isinstance(doc, dict):
        doctype = "dict"
        doc = json.dumps(doc, indent=4, cls=NpEncoder).encode("utf-8")
    elif isinstance(doc, list):
        doctype = "list"
        doc = json.dumps(doc, indent=4, cls=NpEncoder).encode("utf-8")
    elif isinstance(doc, str):
        try:
            doc = json.loads(doc).encode("utf-8")
            doctype = "json"
        except:
            doc = doc.encode("utf-8")
    try:
        if isinstance(doc, pd.DataFrame):
            doc.to_parquet(docpath)
        else:
            print("Uploading file. Is that right?")
            s3.upload_fileobj(io.BytesIO(doc), Bucket=bucket_name, Key=prefix)

        print(f"Saved doc as {doctype} to {docpath}")
        return {doctype: docpath}
    except Exception as e:
        print(e)
        return {doctype: ""}


def delete_doc(path):
    # Create an S3 client
    s3 = boto3.client("s3")
    bucket_name, prefix = bucket_key_from_docpath(path)
    # Delete the object
    s3.delete_object(Bucket=bucket_name, Key=prefix)

    print(f"Object {prefix} was successfully deleted from bucket {bucket_name}.")


def mark_completed(current_path, doc, delete_original=True):
    if "review_folder" in current_path:
        new_path = str(current_path).replace("/review_folder", "/output_folder")
        archive_path = str(current_path).replace("/review_folder", "/archive_folder")
    else:
        new_path = str(current_path).replace("/input_folder", "/output_folder")
        archive_path = str(current_path).replace("/input_folder", "/archive_folder")
    print(current_path)
    print(new_path)
    if new_path != current_path:
        save_output = save_doc(new_path, doc, dated=False)
        copy_doc(current_path, archive_path)
        if (
            delete_original
            and doc_exists(new_path)
            and doc_exists(archive_path)
            and archive_path != current_path
        ):
            delete_doc(current_path)
        return save_output


def list_folders(docpath: str, delimiter: str = "/") -> list:
    """
    List folders in an S3 bucket.

    Parameters:
    - bucket_name: Name of the S3 bucket.
    - prefix: Filter folders starting with this prefix. Defaults to empty string, listing from the bucket root.
    - delimiter: Delimiter to use for identifying folders. Defaults to '/'.

    Returns:
    - A list of folder names (prefixes) within the specified bucket.
    """

    s3 = s3fs.S3FileSystem(anon=False, default_cache_type="none")

    folders = []
    try:
        folders = s3.ls(docpath, detail=False)
    except:
        pass

    return folders


def get_textracted_document(
    docpath, output="", always_extract=False, region_name="ap-southeast-2"
):
    extractor = Textractor(region_name=region_name)
    print(extractor.__dict__)
    features = []
    docname = docpath.split("/")[-1]
    textracted_path = f"{'/'.join(docpath.split('/')[:-2])}/textracted_folder/{docname}"

    if not always_extract and doc_exists(textracted_path):
        print("Reading existing textracted document")
        document = read_doc(textracted_path)
        return document
    analyse = False
    if "table" in output:
        features.append(TextractFeatures.TABLES)
        analyse = True
    elif "form" in output:
        features.append(TextractFeatures.FORMS)
        analyse = True
    elif "layout" in output:
        features.append(TextractFeatures.LAYOUT)
        analyse = True

    if analyse:
        document = extractor.start_document_analysis(
            file_source=docpath,
            features=features,
            save_image=False,
            s3_output_path=textracted_path,
        )
    else:
        document = extractor.start_document_text_detection(
            file_source=docpath, save_image=False, s3_output_path=textracted_path
        )

    save_doc(textracted_path, document.response)
    print(f"Textracted document saved to {textracted_path}")
    return document.response


class SFTPConfig(BaseModel):
    url: str
    username: str
    password: str
    port: int
    from_customer_folder: str


def authenticate_sftp(sftp_config: SFTPConfig):
    cnopts = pysftp.CnOpts()
    cnopts.hostkeys = None
    return pysftp.Connection(
        host=sftp_config["url"],
        username=sftp_config["username"],
        password=sftp_config["password"],
        port=sftp_config["port"],
        cnopts=cnopts,
    )


def fetch_files_from_sftp(sftp) -> List[str]:
    return [file for file in sftp.listdir_attr() if sftp.isfile(file.filename)]


def save_file_to_s3(sftp, file_name: str, local_path: str, s3_path: str, encode=True):
    sftp.get(file_name, local_path)
    doc = Path(local_path).read_text()
    put_doc(doc, s3_path, content_type="text/plain", encode=encode)


def sftp_to_s3(
    sftp_config,
    metadata,
    dated: bool = True,
    encode=True,
    delete=True,
    archive_folder="",
):
    doc_date = ""
    if dated:
        doc_date = arrow.now(metadata.timezone).format("YYYYMMDDHHmm")
        doc_date = f"{doc_date}_"
    sftp = authenticate_sftp(sftp_config)
    sftp.cwd(sftp_config["from_customer_folder"])
    files = fetch_files_from_sftp(sftp)
    for file in files[:]:
        print()
        print(bbcore.function_location)
        print(file.filename)
        local_path = f"{bbcore.temp_path}/{doc_date}{file.filename}"
        s3_path = f"{metadata.input_path}/{doc_date}{bbcore.snake(file.filename)}"

        save_file_to_s3(sftp, file.filename, local_path, s3_path, encode=encode)
        print(f"Saved to {s3_path}")
        if delete and doc_exists(s3_path):
            if archive_folder:
                archive_path = f"../{archive_folder}/{file.filename}"
                try:
                    sftp.rename(file.filename, archive_path)
                    print(f"Moved to {archive_path}")
                except:
                    sftp.rename(file.filename, f"{archive_path}a")
                    print(f"Moved to {archive_path}")
            else:
                sftp.remove(file.filename)
    sftp.close()


class FTPConfig(BaseModel):
    url: str
    username: str
    password: str
    port: int
    from_customer_folder: str
    archive_folder: str


def authenticate_ftp(ftp_config: FTPConfig):
    ftp = ftplib.FTP()
    ftp.connect(ftp_config["url"], ftp_config["port"])
    ftp.login(ftp_config["username"], ftp_config["password"])
    return ftp


def fetch_files_from_ftp(ftp, folder: str, skip_list) -> List[str]:
    ftp.cwd(folder)
    return [filename for filename in ftp.nlst() if filename not in skip_list]


def save_file_to_s3_via_ftp(ftp, filename: str, local_path: str, s3_path: str):
    with open(local_path, "wb") as f:
        ftp.retrbinary(f"RETR {filename}", f.write)
    upload_doc(local_path, s3_path)


def ftp_to_s3(ftp_config: FTPConfig, metadata, skip_list=[], dated: bool = True):
    doc_date = arrow.now(metadata.timezone).format("YYYYMMDDHHmm")
    ftp = authenticate_ftp(ftp_config)

    filenames = fetch_files_from_ftp(
        ftp, ftp_config["from_customer_folder"], skip_list=skip_list
    )
    print(filenames)
    for filename in filenames:
        print()
        print(filename)
        local_path = f"{bbcore.temp_path}/{doc_date}_{filename}"
        s3_path = f"{metadata.input_path}/{doc_date}_{filename}"

        save_file_to_s3_via_ftp(ftp, filename, local_path, s3_path)

        if doc_exists(s3_path):
            ftp.delete(filename)

    ftp.close()


def get_fs():
    if not hasattr(get_fs, "fs"):
        get_fs.fs = s3fs.S3FileSystem()
    return get_fs.fs


def determine_file_type_s3(s3_path):
    """
    Reads a file from an S3 path and uses Magika to determine the file type.

    :param s3_path: The S3 path to the file (e.g., "s3://bucket-name/path/to/file")
    :return: The detected file type or None if unable to determine.
    """
    # Initialize S3FS
    fs = s3fs.S3FileSystem(anon=False)  # Use anon=False if credentials are required

    # Initialize Magika
    m = Magika()

    # Read the file content
    with fs.open(s3_path, "rb") as f:
        file_bytes = f.read()

    # Use Magika to identify the file type
    result = m.identify_bytes(file_bytes)

    # Assuming result has a property or method to get the file type, adjust based on actual implementation
    file_type = result.output.ct_label if result else "Unknown"

    return file_type


def upload_doc(local_path, docpath):
    # Create an S3 client
    s3 = boto3.client("s3")
    bucket_name, prefix = bucket_key_from_docpath(docpath)
    # Delete the object
    s3.upload_file(local_path, Bucket=bucket_name, Key=prefix)
    print(f"Object {prefix} was successfully uploaded to {docpath}.")


# def put_doc(
#     doc, docpath, content_type="", content_disposition="attachment", encode=True
# ):
#     if encode:
#         try:
#             doc = base64.b64decode(doc)
#         except Exception:
#             try:
#                 doc = doc.encode("utf-8")
#             except Exception as e:
#                 print("Not base64 encoded or utf-8 encoded")
#                 print(e)
#                 pass

#     if not content_type:
#         guessed_type, _ = mimetypes.guess_type(docpath[1])
#         content_type = guessed_type or "application/octet-stream"

#     metadata = {"ContentType": content_type, "ContentDisposition": content_disposition}
#     fs = get_fs()
#     if encode:
#         with fs.open(docpath, "wb", **metadata) as f:
#             f.write(doc)
#     else:
#         with fs.open(docpath, "w", **metadata) as f:
#             f.write(doc)

#     info = fs.info(docpath)
#     object_size = info.get("size", 0)
#     if not object_size:
#         raise Exception(f"Failed to save doc to {docpath}")
#     else:
#         print(f"Saved doc to {docpath}")


def determine_content_type_and_disposition(file_content):
    magika = Magika()
    result = magika.identify_bytes(file_content)
    mime_type = result.output.mime_type
    disposition = "inline" if mime_type == "application/pdf" else "attachment"
    return mime_type, disposition


def put_doc(file_object, docpath):
    bucket_name, object_name = bucket_key_from_docpath(docpath)
    try:
        file_content = file_object.read()
    except:
        file_content = file_object

    content_type, disposition = determine_content_type_and_disposition(file_content)

    s3 = boto3.client("s3")
    s3.upload_fileobj(
        Fileobj=BytesIO(file_content),
        Bucket=bucket_name,
        Key=object_name,
        ExtraArgs={
            "ContentType": content_type,
            "ContentDisposition": disposition,
        },
    )
    file_object.seek(0)  # In case the file_object needs to be reused


def generate_presigned_url(docpath, expiration=31536000):
    bucket_name, object_name = bucket_key_from_docpath(docpath)
    s3_client = boto3.client("s3")
    return s3_client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket_name, "Key": object_name},
        ExpiresIn=expiration,
    )


def update_metadata_and_generate_url(docpath):
    s3 = boto3.client("s3")
    bucket_name, object_name = bucket_key_from_docpath(docpath)
    response = s3.get_object(Bucket=bucket_name, Key=object_name)
    file_content = response["Body"].read()

    # Re-use the upload function to update the object with correct metadata
    put_doc(BytesIO(file_content), docpath)

    return generate_presigned_url(docpath)
