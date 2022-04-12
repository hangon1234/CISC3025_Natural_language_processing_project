from django.shortcuts import render
from django.views import View

class NERView(View):
    def get(self, request):
        return render(request, "named_entity_recognition/form.html")

    def post(self, request):
        textarea = request.POST['textarea']
        classification_result = textarea

        return render(request, "named_entity_recognition/ner_result.html",
                      {"classification_result": classification_result,
                       "textarea_input": textarea})