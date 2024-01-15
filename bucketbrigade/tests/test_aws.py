# test_aws.py

from datetime import datetime, timezone
from unittest.mock import patch


from bucketbrigade.aws import (
    bucket_key_from_docpath,
    get_docpaths_to_process,
    get_output_docnames,
    list_docs,
)


def test_bucket_key_from_docpath():
    # Test with a standard path
    assert bucket_key_from_docpath("s3://my-bucket/my-folder/my-file.txt") == (
        "my-bucket",
        "my-folder/my-file.txt",
    )

    # Test with no prefix
    assert bucket_key_from_docpath("s3://my-bucket/") == ("my-bucket", "")

    # Test with a deeply nested path
    assert bucket_key_from_docpath("s3://my-bucket/folder1/folder2/file.txt") == (
        "my-bucket",
        "folder1/folder2/file.txt",
    )


@patch("bucketbrigade.aws.boto3.client")
def test_list_docs(mock_s3_client):
    # Mock S3 response
    mock_s3_client.return_value.get_paginator.return_value.paginate.return_value = [
        {
            "Contents": [
                {
                    "Key": "my-folder/my-file-1.txt",
                    "LastModified": datetime(2023, 1, 15, tzinfo=timezone.utc),
                },
                {
                    "Key": "my-folder/my-file-2.txt",
                    "LastModified": datetime(2023, 1, 20, tzinfo=timezone.utc),
                },
            ]
        }
    ]

    # Test filtering by dates
    result = list_docs("s3://my-bucket/my-folder", "2023-01-10", "2023-01-18")
    assert result == ["s3://my-bucket/my-folder/my-file-1.txt"]

    # Test with no date filters
    result = list_docs("s3://my-bucket/my-folder")
    assert result == [
        "s3://my-bucket/my-folder/my-file-1.txt",
        "s3://my-bucket/my-folder/my-file-2.txt",
    ]


@patch("bucketbrigade.aws.list_docs")
def test_get_output_docnames(mock_list_docs):
    # Mocking list_docs to return specific file paths
    mock_list_docs.side_effect = (
        lambda output_path, include_string="": [
            f"{output_path}/doc1.txt",
            f"{output_path}/doc2.txt",
        ]
        if include_string in output_path
        else []
    )

    parent_folder = "s3://my-bucket/my-input-folder"

    # Test with no include_string
    expected_docs = ["doc1.txt", "doc2.txt", "doc1.txt", "doc2.txt"]
    assert get_output_docnames(parent_folder) == expected_docs

    # Test with include_string
    include_str = "skip_folder"
    expected_docs = ["doc1.txt", "doc2.txt"]
    assert (
        get_output_docnames(parent_folder, include_string=include_str) == expected_docs
    )

    # Test with different folder structure
    parent_folder = "s3://my-bucket/another-folder/my-input-folder"
    expected_docs = ["doc1.txt", "doc2.txt", "doc1.txt", "doc2.txt"]
    assert get_output_docnames(parent_folder) == expected_docs


@patch("bucketbrigade.aws.list_docs")
@patch("bucketbrigade.aws.get_output_docnames")
def test_get_docpaths_to_process(mock_get_output_docnames, mock_list_docs):
    # Mocking list_docs to return specific file paths in the input folder
    mock_list_docs.side_effect = lambda parent_folder, include_string="": [
        f"{parent_folder}/doc1.txt",
        f"{parent_folder}/doc2.txt",
        f"{parent_folder}/doc3.txt",
    ]

    # Revised mocking for get_output_docnames to account for include_string
    def mock_output_docnames(parent_folder, include_string=""):
        if "some_string" in include_string:
            return ["doc2.txt"]
        return ["doc1.txt", "doc2.txt"]

    mock_get_output_docnames.side_effect = mock_output_docnames

    parent_folder = "s3://my-bucket/my-input-folder"

    # Test with no include_string
    expected_docs = ["s3://my-bucket/my-input-folder/doc3.txt"]
    assert get_docpaths_to_process(parent_folder) == expected_docs

    # Test with include_string
    include_str = "some_string"
    expected_docs = [
        "s3://my-bucket/my-input-folder/doc1.txt",
        "s3://my-bucket/my-input-folder/doc3.txt",
    ]
    assert (
        get_docpaths_to_process(parent_folder, include_string=include_str)
        == expected_docs
    )
