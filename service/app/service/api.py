import logging

import connexion
import http
from app.stubs.models.lmc_message_in import LmcMessageIn
from app.stubs.models.lmc_message_out import LmcMessageOut
from .prediction_svc import prediction_service

from shared.util import log_before, log_after

log = logging.getLogger(__name__)


def health_get():
    return http.HTTPStatus.OK


@log_before
@log_after
def predict_api(body):

    """
    retrieve predictions for MM
    :param body: lmc messages
    :type body: list | bytes
    :rtype: Union[LmcMessageOut, Tuple[LmcMessageOut, int], Tuple[LmcMessageOut, int, Dict[str, str]]
    """
    if connexion.request.is_json:
        body = [LmcMessageIn.from_dict(i) for i in body]
    pr_svc = prediction_service()
    res = pr_svc.predict_mappings(body)
    res_dict = [LmcMessageOut.to_dict(i) for i in res]
    return res_dict, http.HTTPStatus.OK
