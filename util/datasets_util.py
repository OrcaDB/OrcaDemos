"""Utility functions for working with datasets in Google Cloud Storage."""

import fnmatch
import glob
import os

import torch
from google.cloud import storage

DEFAULT_BUCKET = "orcadb-datasets-us-east1"
DEFAULT_PROJECT = "orcadb-internal"


def list_files(
    pattern: str,
    bucket_name: str = DEFAULT_BUCKET,
) -> list[str]:
    """List files in a bucket matching a given pattern.

    :param pattern: The pattern to match. Uses fnmatch.
    :param bucket_name: The name of the bucket to list files from.
    """
    storage_client = storage.Client(project=DEFAULT_PROJECT)
    bucket = storage_client.get_bucket(bucket_name)

    blobs = bucket.list_blobs(prefix="", delimiter=None)
    return [blob.name for blob in blobs if fnmatch.fnmatch(blob.name, pattern)]


def download_files_from_bucket(
    pattern: str,
    local_path: str = "temp/",
    bucket_name: str = DEFAULT_BUCKET,
    force: bool = False,
) -> list[str]:
    """Download files from a bucket matching a given pattern.

    :param pattern: The pattern to match. Uses fnmatch.
    :param local_path: The local path to download files to. Defaults to "temp/".
    :param bucket_name: The name of the bucket to download files from.
    :param force: Whether to force download files that already exist locally.
    """
    storage_client = storage.Client(project=DEFAULT_PROJECT)
    bucket = storage_client.get_bucket(bucket_name)

    if not os.path.exists(local_path):
        os.makedirs(local_path)

    blobs = bucket.list_blobs(prefix="", delimiter=None)

    matching_blobs = [blob for blob in blobs if fnmatch.fnmatch(blob.name, pattern)]

    result = []

    for blob in matching_blobs:
        blob_path = blob.name
        file = blob_path.split("/")[-1]
        download_path = os.path.join(local_path, file)

        if os.path.exists(download_path) and not force:
            print(f"File already exists at {download_path} - skipping")
        else:
            print(f"Downloading {blob_path} to {download_path}")
            blob.download_to_filename(download_path)

        result.append(download_path)

    return result


def upload_files_to_bucket(
    local_pattern: str,
    upload_path: str,
    bucket_name: str = DEFAULT_BUCKET,
) -> list[str]:
    """Upload files to a bucket matching a given pattern.

    :param local_pattern: The pattern to match. Uses fnmatch.
    :param upload_path: The path to upload files to in the bucket.
    :param bucket_name: The name of the bucket to upload files to.
    """
    # Initialize the GCS client
    client = storage.Client(project=DEFAULT_PROJECT)

    # Get a reference to the bucket
    bucket = client.get_bucket(bucket_name)

    # Use glob to list files matching the source_path
    local_files = glob.glob(local_pattern)

    upload_paths = []

    # Upload each matching file to the GCS bucket
    for local_file_path in local_files:
        if os.path.isfile(local_file_path):
            filename = os.path.basename(local_file_path)
            upload_filename = os.path.join(upload_path, filename)
            print(f"Uploading {local_file_path} to {upload_filename} in bucket {bucket_name}")
            blob = bucket.blob(upload_filename)
            blob.upload_from_filename(local_file_path)
            upload_paths.append(upload_filename)

    return upload_paths


# Function to calculate mean and std
def calculate_mean_std(loader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0

    for images, _ in loader:
        # Use the .to(device) to perform the computation on GPU if available
        channels_sum += torch.mean(images, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(images**2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches

    # Var[X] = E[X^2] - (E[X])^2
    std = (channels_squared_sum / num_batches - mean**2) ** 0.5

    return mean, std
