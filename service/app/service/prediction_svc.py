from typing import List
import logging
from threading import Lock


from stubs.models.lmc_message_in import LmcMessageIn
from stubs.models.lmc_message_out import LmcMessageOut
from stubs.models.shipment_status_prediction import ShipmentStatusPrediction
from stubs.models.event_type_prediction import EventTypePrediction
from shared.util import log_before, log_after
from .config import *
from prediction.status import StatusPredictor
from prediction.events import EventTypesPredictor

log = logging.getLogger(__name__)


def _build_rest_response(message, slugs, status, events, _return):
    status = [ShipmentStatusPrediction(shipment_status=i, probability=j) for i, j in zip(*status)]
    events_m = []
    for i in zip(*events):
        ev = [EventTypePrediction(event_type=e, probability=p) for e, p in zip(*i)]
        events_m.append(ev)

    res = [
        LmcMessageOut(lmc_message=m, carrier_slug=sl,
                      shipment_status_prediction=st,
                      event_types_prediction=e, is_return=_return)
        for m, sl, st, e in zip(message, slugs, status, events_m)
    ]
    return res


class PredictionService:
    _status_predictor: StatusPredictor
    _return_status_predictor: StatusPredictor
    _event_types_predictor: EventTypesPredictor
    _returns_event_types_predictor: EventTypesPredictor
    mutex = Lock()

    def __init__(self):
        self._status_predictor = StatusPredictor(model_name=SHIPMENT_STATUS_MODEL_DIR)
        self._return_status_predictor = StatusPredictor(model_name=RETURN_SHIPMENT_STATUS_MODEL_DIR)
        self._event_types_predictor = EventTypesPredictor(model_name=EVENT_TYPES_MODEL_DIR)
        self._returns_event_types_predictor = EventTypesPredictor(model_name=RETURN_EVENT_TYPES_MODEL_DIR)

    @log_before
    @log_after
    def predict_mappings(self, req_body: List[LmcMessageIn]) -> List[LmcMessageOut]:
        self.mutex.acquire()

        outbound_messages = []
        outbound_slugs = []
        return_messages = []
        return_slugs = []

        for i in req_body:
            if i.is_return:
                return_messages.append(i.message)
                return_slugs.append(i.carrier_slug)
            else:
                outbound_messages.append(i.message)
                outbound_slugs.append(i.carrier_slug)

        res = []
        if len(outbound_messages) > 0:
            outbound_status = self._status_predictor.predict_batch(outbound_messages)
            outbound_events = self._event_types_predictor.predict_batch(outbound_messages)

            res_outbound = _build_rest_response(outbound_messages, outbound_slugs, outbound_status, outbound_events,
                                                False)
            res += res_outbound

        if len(return_messages) > 0:
            return_status = self._return_status_predictor.predict_batch(return_messages)
            return_events = self._returns_event_types_predictor.predict_batch(return_messages)

            res_return = _build_rest_response(return_messages, return_slugs, return_status, return_events, True)
            res += res_return

        self.mutex.release()
        return res


prediction_svc = PredictionService()

