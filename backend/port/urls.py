from django.urls import path

from backend.port.web.frd_controller import FrdController


urlpatterns = [
    path('frd/', FrdController.as_view(), name='frd'),
]
