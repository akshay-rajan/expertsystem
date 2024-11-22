from django.shortcuts import render

def introduction(request):
    return render(request, 'learn/introduction.html')
