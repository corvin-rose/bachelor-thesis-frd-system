from typing import Callable
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.request import Request


class BaseController(APIView):
    def execute_secured(self, request: Request, callback: Callable[[], Response]):
        if request.headers.get("Api-Key") == "m9JMDj5h8MJMzoKPcRRyfkjC":
            return Response(callback)

        return Response('No valid api key', 403)
