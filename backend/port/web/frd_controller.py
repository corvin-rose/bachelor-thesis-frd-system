from backend.core.domain.service.frd_service import FrdService
from backend.port.web.base_controller import BaseController


class FrdController(BaseController):
    def __init__(self):
        self.service = FrdService()

    def post(self, request):
        return self.execute_secured(request, lambda: self.service.classify(request.data))
