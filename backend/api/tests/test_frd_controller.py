import json
from django.test import TestCase
from django.urls import reverse
from rest_framework.test import APIClient
from unittest.mock import MagicMock, patch

from backend.port.web.base_controller import API_KEY


class FrdControllerTestCase(TestCase):
    def setUp(self):
        # https://www.django-rest-framework.org/api-guide/testing/
        self.client = APIClient()
        self.client.credentials(HTTP_API_KEY=API_KEY)
        self.service_mock = MagicMock()

    # https://docs.python.org/3/library/unittest.mock.html
    @patch('backend.core.domain.service.frd_service.FrdService.classify', return_value={})
    def test_classify_post_method(self, service_mock):
        # given
        data = json.dumps('Test review')

        # when
        response = self.client.post(reverse('frd'), data, format='json')

        # then
        self.assertEqual(response.status_code, 200)
        service_mock.assert_called_once_with(data)
