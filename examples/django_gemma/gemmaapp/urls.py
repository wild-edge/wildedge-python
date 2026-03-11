from django.urls import path

from gemmaapp import views

urlpatterns = [
    path("infer/", views.infer),
]
