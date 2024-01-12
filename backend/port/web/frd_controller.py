from backend.core.domain.service.frd_service import FRDService
from backend.port.web.base_controller import BaseController
from rest_framework.response import Response


class FRDController(BaseController):
    def __init__(self):
        self.__service = FRDService()

    def get(self, request):
        text = "Love this!  Well made, sturdy, and very comfortable.  I love it!Very pretty"
        return self.execute_secured(request, self.__service.classify(text))

    def post(self, request):
        return Response({"message": "POST-Anfrage erhalten"})
