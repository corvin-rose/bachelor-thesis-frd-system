from django.urls import path

from backend.port.web.frd_controller import FRDController


urlpatterns = [
    path('frd/', FRDController.as_view(), name='frd'),
]
