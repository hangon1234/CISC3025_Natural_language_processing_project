from django.shortcuts import render


# Create your views here.
def index(request):
    return render(request, "named_entity_recognition/form.html")
