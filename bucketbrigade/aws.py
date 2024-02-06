import base64
import csv
import io
import json
import polars as pl
from datetime import date, datetime, timezone
from xml.etree import ElementTree as ET

import arrow
import boto3
import dateutil
import numpy as np
import pandas as pd
from botocore.exceptions import ClientError

from bucketbrigade import core as bbcore


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
        return {'bytes': docpath}
    try:
        doctype = "ET"
        doc = ET.tostring(doc, encoding="unicode")
    except:
        pass

    doctype = "text"
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
            doc.to_parquet(docpath, engine="fastparquet")
        else:
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
    new_path = str(current_path).replace("/input_folder", "/output_folder")
    archive_path = str(current_path).replace("/input_folder", "/archive_folder")
    print(current_path)
    print(new_path)
    if new_path != current_path:
        save_output = save_doc(new_path, doc, dated=False)
        copy_doc(current_path, archive_path)
        if delete_original and doc_exists(new_path) and doc_exists(archive_path):
            delete_doc(current_path)
        return save_output


def read_parquet_list(s3_directory, filter_expr=None):
    source = f"{s3_directory}/*"
    lazy_query = pl.scan_parquet(source)
    if filter_expr is not None:
        lazy_query = lazy_query.filter(filter_expr)
    df = lazy_query.collect()
    return df
