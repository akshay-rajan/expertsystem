import os
import uuid
import pickle
import numpy as np
import pandas as pd
from django.conf import settings
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


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
    Display the results and allow the user to download the model
    """
    # On submission of the datasets
    if request.method == 'POST':
        # ! Data Processing
        dataset = request.FILES.get('dataset', None)
        
        file_extension = dataset.name.split('.')[-1]
        if file_extension == 'csv':
            df = pd.read_csv(dataset, )
        else:
            df = pd.read_excel(dataset)
        
        features = [s.replace('\n', '').replace('\r', '') for s in request.POST.getlist('features')]
        target = request.POST.get('target').replace('\n', '').replace('\r', '')
                
        # Features and Target selection
        X = df[features]
        y = df[target]

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # ! Model Building
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # ! Model Evaluation
        y_pred = model.predict(X_test)
        y_pred_modified = [round(i, 3) for i in y_pred]
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        b0 = round(model.intercept_, 3)
        b1 = round(model.coef_[0], 3)
        line = f"y = {b0} + {b1}x"
        
        # Serialize the model
        model_filename = f"linear_regression_{uuid.uuid4().hex[:6]}.pkl"
        model_path = os.path.join(settings.MEDIA_ROOT, model_filename)
        
        with open(model_path, 'wb') as file:
            pickle.dump(model, file)
        
        # Provide a download link
        download_link = os.path.join(settings.MEDIA_URL, model_filename)
        
        return render(request, 'main/linear_regression.html', {
            'actual': y_test,
            'predicted': y_pred_modified,
            'metrics': {
                'mse': round(mse, 2),
                'rmse': round(rmse, 2),
                'mae': round(mae, 2),
                'r2': round(r2, 2),
            },
            'line': line,
            'download': download_link,
        })
        
    
    # Render the Input Form
    return render(request, 'main/input.html')
    
def knn(request):
    """Build KNN model and evaluate it"""
    
    if request.method == "POST":
        dataset = request.FILES.get('dataset', None)
        
        file_extension = dataset.name.split('.')[-1]
        if file_extension == 'csv':
            df = pd.read_csv(dataset, )
        else:
            df = pd.read_excel(dataset)
        
        features = [s.replace('\n', '').replace('\r', '') for s in request.POST.getlist('features')]
        target = request.POST.get('target').replace('\n', '').replace('\r', '')
                
        # Features and Target selection
        X = df[features]
        y = df[target]

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # ! Model Building
        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(X_train, y_train)
        
        # ! Model Evaluation
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        target_names = [str(x) for x in df[target].unique().tolist()] # Not the field names, since data is encoded
        report = classification_report(y_test, y_pred, target_names=target_names)
        
        model_filename = f"knn_{uuid.uuid4().hex[:6]}.pkl"
        model_path = os.path.join(settings.MEDIA_ROOT, model_filename)
        
        with open(model_path, 'wb') as file:
            pickle.dump(model, file)
        
        download_link = os.path.join(settings.MEDIA_URL, model_filename)
        
        return render(request, 'main/knn.html', {
            'actual': y_test,
            'predicted': y_pred,
            'metrics': {
                'accuracy': round(accuracy, 2),
                'precision': round(precision, 2),
                'recall': round(recall, 2),
                'f1': round(f1, 2),
            },
            'download': download_link,
        })
    return render(request, 'main/input.html')
    
def samples(request):
    return render(request, 'main/samples.html')

# ? Helper Functions

