from rest_framework import serializers

from backend.core.domain.model.authenticity import Authenticity


class ClassificationResult:
    def __init__(self, input_text: str, result: Authenticity, probability: float):
        self.input_text = input_text
        self.result = result
        self.probability = probability


class ClassificationResultSerializer(serializers.Serializer):
    input_text = serializers.CharField()
    result = serializers.CharField()
    probability = serializers.FloatField()
