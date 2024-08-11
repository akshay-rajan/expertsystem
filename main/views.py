from django.shortcuts import render, redirect
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from django.http import JsonResponse
import numpy as np
import pandas as pd


def index(request):
    return render(request, 'main/index.html')

def classification(request):
    return render(request, 'main/classification.html')

def regression(request):
    return render(request, 'main/regression.html')
    
def clustering(request):
    return render(request, 'main/clustering.html')

def todo(request):
    """
    Render the home page. 
    Let user select a dataset.
    Redirect to the page for selection of variables.
    """

    datasets = {
        'fetch_california_housing': fetch_california_housing,   
    }

    if request.method == 'POST':
        dataset_name = request.POST.get('dataset')
        indep_vars = request.POST.getlist('independent_vars')
        dep_var = request.POST.get('dependent_var')
        
        # Load dataset
        dataset_func = datasets[dataset_name]
        dataset = dataset_func()
        df = pd.DataFrame(data=dataset.data, columns=dataset.feature_names)
        df['target'] = dataset.target

        X = df[indep_vars]
        y = df[dep_var]

        # Split the dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        lr = LinearRegression()
        lr.fit(X_train, y_train)

        # Predict the test values
        predictions = lr.predict(X_test)

        # Prepare the results
        results = {
            'actual': y_test.tolist(),
            'predicted': predictions.tolist()
        }

        return JsonResponse(results)
    else:
        return render(request, 'main/index.html', {'datasets': datasets.keys()})

def get_variables(request):
    dataset_name = request.GET.get('dataset')
    datasets = {
        'fetch_california_housing': fetch_california_housing,
    }
    dataset_func = datasets[dataset_name]
    dataset = dataset_func()
    variables = dataset.feature_names
    return JsonResponse({'variables': variables.tolist()})
