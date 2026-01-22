"""
script: helpers.py

Utility functions for the project.
"""

# Standard Python Library imports
import os
from logging import Logger
from pathlib import Path, PurePosixPath

# Standard Library Type-Hinting
from typing import Optional

# Third-party Libraries
import boto3
from sagemaker.serve.spec.inference_spec import InferenceSpec
import torch
import numpy as np

# Third-party Type-Hinting
from boto3.session import Session


def ret_AWS_URI_subcomp(
    logger: Logger, bucket: str, filepath: str
) -> tuple[str, str, str, str, str]:
    """Utility function to help construct the URI for an AWS resource and provide its subcomponents.

    Args:
        logger (Logger): The logger instance to log messages.
        bucket (str): A base for the AWS resource.
        filepath (str): The filepath name of AWS resource.

    Returns:
        aws_uri (str): The AWS resource URI.
        aws_bucket (str): The Bucket containing the resource.
        aws_filename (str): The filename of the resource.
        aws_extension (str): The file type/extension.
    """
    aws_bucket = bucket.replace("s3://", "").split("/")[0]

    # Use PurePosixPath to handle the filepath logic
    # This handles "folder/sub/file.csv" or just "file.csv" identically
    path_obj = PurePosixPath(filepath)

    if ".." in path_obj.parts:
        logger.warning("Relative path '..' detected. Stripping to filename only.")
        path_obj = PurePosixPath(path_obj.name)

    # Extract components
    aws_filename = path_obj.stem  # 'data-RAW'
    aws_extension = path_obj.suffix  # '.csv' (includes the dot)

    # Extract the "filler" (the parent directories)
    # If filepath is "data.csv", parent is "."
    filler = path_obj.parent.as_posix()

    # 5. Construct the URI
    if filler == ".":
        # Case: File is in the root of the bucket
        aws_uri = f"s3://{aws_bucket}/{path_obj.name}"
        filler = None
    else:
        # Case: File is inside subdirectories
        aws_uri = f"s3://{aws_bucket}/{filler}/{path_obj.name}"

    return aws_uri, aws_bucket, aws_filename, aws_extension, filler


def get_uri(logger: Logger, bucket, path):
    """Utility function to construct and return just the URI for an AWS resource.

    Args:
        logger (Logger): The logger instance to log messages.
        bucket (str): A base for the AWS resource.
        filepath (str): The filepath name of AWS resource.

    Returns:
        aws_uri (str): The AWS resource URI.
    """
    return ret_AWS_URI_subcomp(logger, bucket, path)[0]


def make_boto3_session(logger: Logger) -> Session:
    """Returns the current boto3 session. Creates one if there is no existing session yet.

    Args:
        logger (Logger): The logger instance to log messages.

    Returns:
        (Session): An active boto3 Session.
    """
    # Prefer explicit region; fallback to env/default chain
    region = os.getenv("AWS_REGION", "us-east-1")
    profile = os.getenv("AWS_PROFILE", None)  # only effective if ~/.aws is mounted
    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID", None)
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY", None)

    if profile:
        logger.info("Creating boto3 Session with profile=%s region=%s", profile, region)
        return boto3.Session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            profile_name=profile,
            region_name=region,
        )
        # requires mounted ~/.aws with valid [profile <name>] in config and [<name>] in credentials
    else:
        logger.info("Creating boto3 default Session region=%s", region)
        return boto3.Session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region,
        )


def check_aws_creds(logger: Logger, session=None) -> Session:
    """Checks for current AWS credentials and retrieves a session.

    Args:
        logger (Logger): The logger instance to log messages.

    Returns:
        (Session): An active boto3 Session.
    """
    if session is None:
        session = make_boto3_session(logger)
    creds = session.get_credentials()
    if not creds:
        raise RuntimeError("AWS credentials not found. Check your environment or IAM Role.")
    frozen = creds.get_frozen_credentials()
    logger.info("AWS creds detected: access_key_id starts with %s***", frozen.access_key[:4])
    return session
