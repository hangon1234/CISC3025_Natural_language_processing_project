from django.urls import path

from . import views

app_name = 'text_classification'

urlpatterns = [
    path('', views.index, name='index'),
]