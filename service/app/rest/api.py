import logging

import connexion
import http

from service.prediction_svc import prediction_svc
from ..stubs.models.lmc_message_in import LmcMessageIn
from ..stubs.models.lmc_message_out import LmcMessageOut

from shared.util import log_before, log_after

log = logging.getLogger(__name__)


def health_get():
    return http.HTTPStatus.OK


@log_before
@log_after
def predict_api(body: LmcMessageIn):
    """retrieve predictions for MM
    :param body: lmc messages
    :type body: list | bytes
    :rtype: Union[LmcMessageOut, Tuple[LmcMessageOut, int], Tuple[LmcMessageOut, int, Dict[str, str]]
    """
    if connexion.request.is_json:
        body = [LmcMessageIn.from_dict(d) for d in connexion.request.get_json()]  # noqa: E501
    res = prediction_svc.predict_mappings(body)
    res_dict = [LmcMessageOut.to_dict(i) for i in res]
    return res_dict, http.HTTPStatus.OK
