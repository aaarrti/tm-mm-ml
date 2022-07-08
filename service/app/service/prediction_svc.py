from typing import List
import logging
from threading import Lock
from functools import lru_cache


from app.stubs.models.lmc_message_in import LmcMessageIn
from app.stubs.models.lmc_message_out import LmcMessageOut
from app.stubs.models.shipment_status_prediction import ShipmentStatusPrediction
from app.stubs.models.event_type_prediction import EventTypePrediction
from shared import log_before, log_after
from app.prediction.status import StatusPredictor
from app.prediction.events import EventTypesPredictor
from app.config import SHIPMENT_STATUS_MODEL_TAG, RETURN_SHIPMENT_STATUS_MODEL_TAG, EVENT_TYPES_MODEL_TAG, RETURN_EVENT_TYPES_MODEL_TAG

log = logging.getLogger(__name__)


def _build_rest_response(message, slugs, status, events, is_return):
    status = [ShipmentStatusPrediction(shipment_status=i, probability=j) for i, j in zip(*status)]
    events_m = []
    for i in zip(*events):
        ev = [EventTypePrediction(event_type=e, probability=p) for e, p in zip(*i)]
        events_m.append(ev)

    res = [
        LmcMessageOut(lmc_message=m, carrier_slug=sl,
                      shipment_status_prediction=st,
                      event_types_prediction=e, is_return=is_return)
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
        self._status_predictor = StatusPredictor(SHIPMENT_STATUS_MODEL_TAG)
        self._return_status_predictor = StatusPredictor(RETURN_SHIPMENT_STATUS_MODEL_TAG)
        self._event_types_predictor = EventTypesPredictor(EVENT_TYPES_MODEL_TAG)
        self._returns_event_types_predictor = EventTypesPredictor(RETURN_EVENT_TYPES_MODEL_TAG)

    # @log_before
    # @log_after
    def predict_mappings(self, req_body: List[LmcMessageIn]) -> List[LmcMessageOut]:
        self.mutex.acquire()
        mapping = self._predict_mappings(req_body)
        self.mutex.release()
        return mapping

    def _predict_mappings(self, req_body: List[LmcMessageIn]) -> List[LmcMessageOut]:
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

            res_outbound = _build_rest_response(message=outbound_messages,
                                                slugs=outbound_slugs,
                                                status=outbound_status,
                                                events=outbound_events,
                                                is_return=False
                                                )
            res += res_outbound

        if len(return_messages) > 0:
            return_status = self._return_status_predictor.predict_batch(return_messages)
            return_events = self._returns_event_types_predictor.predict_batch(return_messages)

            res_return = _build_rest_response(message=return_messages,
                                              slugs=return_slugs,
                                              status=return_status,
                                              events=return_events,
                                              is_return=True
                                              )
            res += res_return

        return res


@lru_cache(maxsize=None)
def prediction_service() -> PredictionService:
    return PredictionService()
