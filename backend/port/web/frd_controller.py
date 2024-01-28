from backend.core.domain.service.frd_service import FRDService
from backend.port.web.base_controller import BaseController


class FRDController(BaseController):
    def __init__(self):
        self.service = FRDService()

    def post(self, request):
        return self.execute_secured(request, lambda: self.service.classify(request.data))
