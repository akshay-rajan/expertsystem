from django.shortcuts import render, redirect
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from django.http import JsonResponse
import numpy as np
import pandas as pd


def index(request):
    return render(request, 'main/index.html')

# ? Types of Machine Learning Algorithms

def classification(request):
    return render(request, 'main/classification.html')

def regression(request):
    return render(request, 'main/regression.html')
    
def clustering(request):
    return render(request, 'main/clustering.html')

# ? Implement Algorithms

def linear_regression(request):
    """
    Enable user to input training and testing sets
    Build a Linear Regression model
    Display the results
    """    
    # On submission of the datasets
    if request.method == 'POST':
        # ! Data Processing
        # train_set = request.POST.get('train_set')
        # test_set = request.POST.get('test_set')
        
        # Load the datasets
        # train_set = load_dataset(train_set)
        # test_set = load_dataset(test_set)
        
        data = {
            'Size': [1500, 1600, 1700, 1800, 1900],
            'Price': [300, 320, 340, 360, 380]
        }
        df = pd.DataFrame(data)
        print(df.head())
                
        # Features, Target
        # X, y = df['Size'], df['Price']
        X = np.array(df['Size']).reshape(-1, 1)
        y = np.array(df['Price'])
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # ! Model Building
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # ! Model Evaluation
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        b0 = model.intercept_
        b1 = model.coef_[0]
        line = f"y = {b0} + {b1}x"
        print(line)
        
        return render(request, 'main/results.html', {
            'actual': y_test,
            'predicted': y_pred,
            'metrics': {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
            },
            'line': line,
        })
        
    
    # Render the Input Form
    train_set = ["My Train Set"]
    test_set = ["My Test Set"]
    return render(request, 'main/input.html', {
        'train_set': train_set, 
        'test_set': test_set
    })

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
