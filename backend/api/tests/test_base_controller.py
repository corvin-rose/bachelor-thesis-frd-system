from django.test import TestCase
from rest_framework.test import APIRequestFactory

from backend.port.web.base_controller import BaseController, API_KEY


class BaseControllerTestCase(TestCase):
    def setUp(self):
        self.factory = APIRequestFactory()
        self.controller = BaseController()

    def test_execute_secured_with_valid_api_key(self):
        # given
        headers = {'HTTP_API_KEY': API_KEY}     # https://stackoverflow.com/questions/68729084/how-do-i-add-a-header-to-a-django-requestfactory-request
        request = self.factory.post('/frd/', **headers)

        # when
        response = self.controller.execute_secured(request, lambda: 'success')

        # then
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data, 'success')

    def test_execute_secured_with_invalid_api_key(self):
        # given
        headers = {'HTTP_API_KEY': 'invalid-key'}
        request = self.factory.post('/frd/', **headers)

        # when
        response = self.controller.execute_secured(request, lambda: 'success')

        # then
        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.data, 'No valid api key')
