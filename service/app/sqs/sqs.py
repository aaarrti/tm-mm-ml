import logging
import boto3
from .config import *
from shared.aws.auth import auth
from shared.util import log_before, log_after
from typing import List, Dict
import json
from botocore.client import ClientError


log = logging.getLogger(__name__)


def send_messages(queue, messages: List):
    """
    Send a batch of messages in a single request to an SQS queue.
    This request may return overall success even when some messages were not sent.
    The caller must inspect the Successful and Failed lists in the response and
    resend any failed messages.

    :param queue: The queue to receive the messages.
    :param messages: The messages to send to the queue. These are simplified to
                     contain only the message body and attributes.
    :return: The response from SQS that contains the list of successful and failed
             messages.
    """

    outs = [json.dumps(i.to_dict()) for i in messages]
    while len(outs) > 0:
        batch = outs[:SQS_BATCH_SIZE]
        try:
            entries = [{
                'Id': str(ind),
                'MessageBody': msg,
                'MessageAttributes': SQS_MESSAGE_ATTRIBUTES
            } for ind, msg in enumerate(batch)]
            response = queue.send_messages(Entries=entries)

            if 'Successful' in response:
                for msg_meta in response['Successful']:
                    log.info(
                        "Message sent: %s: %s",
                        msg_meta['MessageId'],
                        batch[int(msg_meta['Id'])]
                    )
            if 'Failed' in response:
                for msg_meta in response['Failed']:
                    log.warning(
                        "Failed to send: %s: %s",
                        msg_meta['MessageId'],
                        batch[int(msg_meta['Id'])]
                    )
        except ClientError as error:
            log.exception("Send messages failed to queue: %s", queue)
            raise error
        else:
            outs = outs[SQS_BATCH_SIZE:]


@log_before
@log_after
def receive_messages(queue) -> List[Dict]:
    """
    Receive a batch of messages in a single request from an SQS queue.

    :param queue: The queue from which to receive messages.
    :return: The list of Message objects received as dict.
    """
    try:
        messages: List = queue.receive_messages(
            MessageAttributeNames=['All'],
            MaxNumberOfMessages=SQS_BATCH_SIZE,
            WaitTimeSeconds=SQS_MESSAGE_WAIT_TIME_SEC
        )
        obj_m = []
        for msg in messages:
            log.info("Received message: %s: %s", msg.message_id, msg.body)
            [obj_m.append(i) for i in json.loads(msg.body)]
            msg.delete()
    except ClientError as error:
        log.exception("Couldn't receive messages from queue: %s", queue)
        raise error
    else:
        return obj_m


def sqs_client():
    if USE_ELASTIC_MQ:
        return boto3.resource('sqs',
                              endpoint_url='http://localhost:9324',
                              region_name='elasticmq',
                              aws_secret_access_key='x',
                              aws_access_key_id='x'
                              )
    else:
        key, secret, token = auth()
        return boto3.resource('sqs',
                              aws_secret_access_key=key,
                              aws_access_key_id=secret,
                              aws_session_token=token
                              )


def in_queue():
    client = sqs_client()
    return client.get_queue_by_name(QueueName=IN_QUEUE_NAME)


def out_queue():
    client = sqs_client()
    return client.get_queue_by_name(QueueName=OUT_QUEUE_NAME)
