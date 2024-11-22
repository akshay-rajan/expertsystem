from django.shortcuts import render

def introduction(request):
    return render(request, 'learn/introduction.html')

def machinelearning(request):
    return render(request, 'learn/machinelearning.html')
