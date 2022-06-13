import boto3
import os
import logging


log = logging.getLogger(__name__)

__all__ = [
    'auth'
]


def auth():
    if 'AWS_WEB_IDENTITY_TOKEN_FILE' in os.environ:
        log.info('AUTH with service account')
        return _auth_with_service_account()
    if 'AWS_ACCESS_KEY_ID' in os.environ:
        log.info('Auth with AWS keys')
        return os.environ['AWS_ACCESS_KEY_ID'], os.environ['AWS_SECRET_ACCESS_KEY'], os.environ['AWS_SESSION_TOKEN']
    raise Exception('No AWS auth')


def _auth_with_service_account():
    log.info('Authenticating with AWS service account')
    with open(os.environ['AWS_WEB_IDENTITY_TOKEN_FILE'], 'r') as content_file:
        web_identity_token = content_file.read()
    role_arn = os.environ['AWS_ROLE_ARN']
    # create an STS client object that represents a live connection to the
    # STS service
    sts_client = boto3.client('sts')
    # Call the assume_role method of the STSConnection object and pass the role
    # ARN and a role session name.
    assumed_role_object = sts_client.assume_role_with_web_identity(
        RoleArn=role_arn,
        RoleSessionName="AssumeRoleSession1",
        WebIdentityToken=web_identity_token
    )
    # From the response that contains the assumed role, get the temporary
    # credentials that can be used to make subsequent API calls
    credentials = assumed_role_object['Credentials']
    return credentials['AccessKeyId'], credentials['SecretAccessKey'], credentials['SessionToken']
