import logging
import os
import boto3

from .auth import auth
from .config import S3_BUCKET_NAME

log = logging.getLogger(__name__)

__all__ = [
    'upload_folder_to_s3',
    'download_dir'
]


def s3_bucket():
    return s3().Bucket(S3_BUCKET_NAME)


def s3():
    acces_key, secret_key, session_id = auth()
    return boto3.client(
        's3',
        aws_access_key_id=acces_key,
        aws_secret_access_key=secret_key,
        aws_session_token=session_id,
    )


def upload_folder_to_s3(inputDir, s3Path):
    s3bucket = s3_bucket()
    log.info("Uploading results to s3 initiated...")
    log.info("Local Source:", inputDir)
    os.system("ls -ltR " + inputDir)

    log.info("Dest  S3path:", s3Path)

    try:
        for path, subdirs, files in os.walk(inputDir):
            for file in files:
                dest_path = path.replace(inputDir, "")
                __s3file = os.path.normpath(s3Path + '/' + dest_path + '/' + file)
                __local_file = os.path.join(path, file)
                print("upload : ", __local_file, " to Target: ", __s3file, end="")
                s3bucket.upload_file(__local_file, __s3file)
                log.debug(" ...Success")
    except Exception as e:
        log.error(" ... Failed!! Quitting Upload!!")
        log.error(e)
        raise e


def download_dir(prefix, local):
    """
    params:
    - prefix: pattern to match in s3
    - local: local path to folder in which to place files
    - bucket: s3 bucket with target contents
    - client: initialized s3 client object
    """
    __s3 = s3()
    keys = []
    dirs = []
    next_token = ''
    base_kwargs = {
        'Bucket': S3_BUCKET_NAME,
        'Prefix': prefix,
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
        __s3.download_file(S3_BUCKET_NAME, k, dest_pathname)
