from django.urls import path

from named_entity_recognition.views import NERView

app_name = 'named_entity_recognition'

urlpatterns = [
    path('', NERView.as_view(), name='index'),
]