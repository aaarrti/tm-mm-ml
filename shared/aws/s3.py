import logging
import os
import boto3
from pathlib import Path

from .auth import auth
from shared.config import S3_BUCKET_NAME
from shared.util import log_before

log = logging.getLogger(__name__)

__all__ = [
    'upload_folder_to_s3',
    'download_folder_from_s3'
]


def s3_client():
    acces_key, secret_key, session_id = auth()
    return boto3.client(
        's3',
        aws_access_key_id=acces_key,
        aws_secret_access_key=secret_key,
        aws_session_token=session_id,
    )


def s3_resource():
    acces_key, secret_key, session_id = auth()
    return boto3.resource(
        's3',
        region_name='eu-central-1',
        aws_access_key_id=acces_key,
        aws_secret_access_key=secret_key,
        aws_session_token=session_id,
    )


@log_before
def upload_folder_to_s3(local, s3Path):
    s3 = s3_client()
    for path, subdirs, files in os.walk(local):
        for file in files:
            dest_path = path.replace(local, "")
            __s3file = os.path.normpath(s3Path + '/' + dest_path + '/' + file)
            __local_file = os.path.join(path, file)
            print("upload : ", __local_file, " to Target: ", __s3file, end="")
            s3.upload_file(__local_file, S3_BUCKET_NAME, __s3file)
            print(" ...Success")


@log_before
def download_folder_from_s3(local, s3_path):
    """
    params:
    - prefix: pattern to match in s3
    - local: local path to folder in which to place files
    - bucket: s3 bucket with target contents
    - client: initialized s3 client object
    """
    __s3 = s3_client()
    keys = []
    dirs = []
    next_token = ''
    base_kwargs = {
        'Bucket': S3_BUCKET_NAME,
        'Prefix': s3_path,
    }
    while next_token is not None:
        kwargs = base_kwargs.copy()
        if next_token != '':
            kwargs.update({'ContinuationToken': next_token})
        results = __s3.list_objects_v2(**kwargs)
        contents = results.get('Contents')
        for i in contents:
            k = i.get('Key')
            if k[-1] != '/':
                keys.append(k)
            else:
                dirs.append(k)
        next_token = results.get('NextContinuationToken')
    for d in dirs:
        dest_pathname = os.path.join(local, d)
        if not os.path.exists(os.path.dirname(dest_pathname)):
            os.makedirs(os.path.dirname(dest_pathname))
    for k in keys:
        dest_pathname = os.path.join(local, k)
        if not os.path.exists(os.path.dirname(dest_pathname)):
            os.makedirs(os.path.dirname(dest_pathname))
        if Path(dest_pathname).exists():
            print(f'{dest_pathname} already exists')
        else:
            print(f'Downloading {dest_pathname}')
            __s3.download_file(S3_BUCKET_NAME, k, dest_pathname)
