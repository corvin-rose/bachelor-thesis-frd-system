from backend.core.domain.model.authenticity import Authenticity
from backend.core.domain.model.classification_result import ClassificationResult, ClassificationResultSerializer
from backend.port.adapter.bert_class import load_bert_model


class FrdService:
    def __init__(self, model=load_bert_model()):
        self._bert_class = model

    def classify(self, text: str):
        # Ermitteln der Wahrscheinlichkeit für fake
        probability = self._bert_class.classify(text)
        # Wahrscheinlichkeit interpretieren (fake oder real)
        result = Authenticity.FAKE if probability >= 0.5 else Authenticity.REAL

        # Falls real, dann Wahrscheinlichkeit invertieren
        if result == Authenticity.REAL:
            probability = 1 - probability

        # Daten für Controller vorbereiten
        classification_result = ClassificationResult(text, result, probability)
        serializer = ClassificationResultSerializer(classification_result)
        return serializer.data
