from django.shortcuts import render
from django.views import View

from starter_codes.NER.MEM import MEMM

class NERView(View):
    def get(self, request):
        return render(request, "named_entity_recognition/form.html")

    def post(self, request):
        textarea = request.POST['textarea']
        classifier = MEMM()

        result = classifier.classify(textarea)
        words = textarea.split()

        for i, word in enumerate(words):
            if result[i] == "PERSON":
                words[i] = f"<mark class='text-danger'>{word}</mark>"

        classification_result = f"<p class='col-md-8 fs-4'>{' '.join(words)}</p>"

        return render(request, "named_entity_recognition/ner_result.html",
                      {"classification_result": classification_result,
                       "textarea_input": textarea})