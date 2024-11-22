from django.shortcuts import render, redirect

TEMPLATES = {
    'introduction': 'learn/introduction.html',
    'machinelearning': 'learn/machinelearning.html',
}

def index(request):
    return redirect('chapter', chapter='introduction')

def chapter_view(request, chapter):
    template = TEMPLATES.get(chapter)
    if template:
        return render(request, template)
    return redirect('introduction/')
