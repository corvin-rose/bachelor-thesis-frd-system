from unittest.mock import MagicMock
from django.test import TestCase

from backend.core.domain.model.authenticity import Authenticity
from backend.core.domain.service.frd_service import FrdService


class FrdServiceTestCase(TestCase):
    def setUp(self):
        self.service = FrdService(model=None)

    def test_classify_real_text(self):
        # given
        fake_probability = 0.02     # Text ist real
        model_mock = MagicMock()
        model_mock.classify.return_value = fake_probability
        self.service._bert_class = model_mock

        # when
        result = self.service.classify("real")

        # then
        self.assertEqual(result.get('probability'), 1 - fake_probability)
        self.assertEqual(result.get('result'), str(Authenticity.REAL))

    def test_classify_fake_text(self):
        # given
        fake_probability = 0.95      # Text ist fake
        model_mock = MagicMock()
        model_mock.classify.return_value = fake_probability
        self.service._bert_class = model_mock

        # when
        result = self.service.classify("fake")

        # then
        self.assertEqual(result.get('probability'), fake_probability)
        self.assertEqual(result.get('result'), str(Authenticity.FAKE))
