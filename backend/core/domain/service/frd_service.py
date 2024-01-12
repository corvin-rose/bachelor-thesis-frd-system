from backend.core.domain.model.authenticity import Authenticity
from backend.core.domain.model.classification_result import ClassificationResult, ClassificationResultSerializer
from backend.port.adapter.bert_class import load_bert_model


class FRDService:
    def __init__(self, model="bert"):
        self.__model = model

    def classify(self, text: str):
        model = load_bert_model()
        probability = model.classify(text)
        result = Authenticity.FAKE if probability >= 0.5 else Authenticity.REAL

        if result == Authenticity.REAL:
            probability = 1 - probability

        classification_result = ClassificationResult(text, result, probability)
        serializer = ClassificationResultSerializer(classification_result)
        return serializer.data
