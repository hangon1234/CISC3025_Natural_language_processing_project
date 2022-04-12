from django.shortcuts import render
from django.views import View

class NERView(View):
    def get(self, request):
        return render(request, "named_entity_recognition/form.html")

    def post(self, request):
        input_text = request.POST['textarea']
        return render(request, "named_entity_recognition/ner_result.html")
# Create your views here.
def index(request):
    return render(request, "named_entity_recognition/form.html")
