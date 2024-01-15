import boto3
from datetime import datetime, timezone
from botocore.exceptions import ClientError


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


def get_output_docnames(parent_folder, include_string="", output_folders=["skip_folder", "output_folder"]):
    """
    Generates a list of document names in specified output folders.

    Args:
    - metadata: An object containing various pieces of information, including the input path.
    - include_string: A string used for filtering documents in the output folders.

    Returns:
    - A list of document names present in the 'skip_folder' and 'output_folder'.
    """
    output_docpaths = []
    input_folder = parent_folder.split('/')[-1]
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
    # Get names of documents in the output folders
    output_docnames = get_output_docnames(parent_folder, include_string=include_string)
    # Filter out documents that are already in the output folders
    return [x for x in input_docpaths if x.split("/")[-1] not in output_docnames]
