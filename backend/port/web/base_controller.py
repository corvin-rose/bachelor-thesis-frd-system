from typing import Callable, Any
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.request import Request

API_KEY = "m9JMDj5h8MJMzoKPcRRyfkjC"


class BaseController(APIView):
    def execute_secured(self, request: Request, callback: Callable[[], Any]):
        if request.headers.get("Api-Key") == API_KEY:
            return Response(callback())

        return Response('No valid api key', 403)
