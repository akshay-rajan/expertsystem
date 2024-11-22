from django.shortcuts import render, redirect

def index(request):
    return redirect('introduction/')

def introduction(request):
    return render(request, 'learn/introduction.html')

def machinelearning(request):
    return render(request, 'learn/machinelearning.html')
