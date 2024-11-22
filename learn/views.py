from django.shortcuts import render, redirect

TEMPLATES = {
    'introduction': 'learn/introduction.html',
    'steps': 'learn/steps.html',
    'preprocessing': 'learn/preprocessing.html',
    'numpy': 'learn/numpy.html',
}

def index(request):
    return redirect('chapter', chapter='introduction')

def chapter_view(request, chapter):
    template = TEMPLATES.get(chapter)
    if template:
        return render(request, template, {'chapters': TEMPLATES})
    return redirect('chapter', 'introduction')
