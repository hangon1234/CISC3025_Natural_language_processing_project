from django.urls import path

from . import views

app_name = 'named_entity_recognition'

urlpatterns = [
    path('', views.index, name='index'),
]