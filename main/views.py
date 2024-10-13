import os
import json
import uuid
import pickle
import numpy as np
import pandas as pd
from django.conf import settings
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, root_mean_squared_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import silhouette_score

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
        rmse = root_mean_squared_error(y_test, y_pred)
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
                
        X = df[features]
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        n_neighbors = int(request.POST.get('n_neighbors'))
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
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
    
    return render(request, 'main/input.html', {
        'hyperparameters': {
            1: {
                'name': 'n_neighbors',
                'type': 'number',
            },
        }
    })

def decision_tree(request):
    """Decision Tree Classifier"""
    
    if request.method == "POST":
        dataset = request.FILES.get('dataset', None)
        
        file_extension = dataset.name.split('.')[-1]
        if file_extension == 'csv':
            df = pd.read_csv(dataset, )
        else:
            df = pd.read_excel(dataset)
        
        features = [s.replace('\n', '').replace('\r', '') for s in request.POST.getlist('features')]
        target = request.POST.get('target').replace('\n', '').replace('\r', '')
                
        X, y = df[features], df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        model_filename = f"decision_tree_{uuid.uuid4().hex[:6]}.pkl"
        model_path = os.path.join(settings.MEDIA_ROOT, model_filename)
        
        download_link = os.path.join(settings.MEDIA_URL, model_filename)
        
        with open(model_path, 'wb') as file:
            pickle.dump(model, file)
            
        plt.figure(figsize=(20,10))
        class_names_str = [str(cls) for cls in model.classes_]
        plot_tree(model, feature_names=features, class_names=class_names_str, filled=True)
        
        plot_filename = f"decision_tree_plot_{uuid.uuid4().hex[:6]}.png"
        plot_path = os.path.join(settings.MEDIA_ROOT, plot_filename)
        plt.savefig(plot_path)
        plt.close()
        plot_url = os.path.join(settings.MEDIA_URL, plot_filename)        
        
        return render(request, 'main/decision_tree.html', {
            'actual': y_test,
            'predicted': y_pred,
            'metrics': {
                'accuracy': round(accuracy, 2),
                'precision': round(precision, 2),
                'recall': round(recall, 2),
                'f1': round(f1, 2),
            },
            'download': download_link,
            'plot': plot_url,
        })
    
    return render(request, 'main/input.html')

def random_forest(request):
    """Random Forest Classifier"""
    
    if request.method == "POST":
        dataset = request.FILES.get('dataset', None)
        file_extension = dataset.name.split('.')[-1]
        if file_extension == 'csv':
            df = pd.read_csv(dataset, )
        else:
            df = pd.read_excel(dataset)
        features = [s.replace('\n', '').replace('\r', '') for s in request.POST.getlist('features')]
        target = request.POST.get('target').replace('\n', '').replace('\r', '')
        n_estimators = int(request.POST.get('n_estimators'))
                
        X, y = df[features], df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        model_filename = f"random_forest_{uuid.uuid4().hex[:6]}.pkl"
        model_path = os.path.join(settings.MEDIA_ROOT, model_filename)
        download_link = os.path.join(settings.MEDIA_URL, model_filename)
        with open(model_path, 'wb') as file:
            pickle.dump(model, file)

        # Plot Feature Importances
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(10,6))
        plt.title("Feature Importances")
        plt.bar(range(X.shape[1]), importances[indices], align="center", color="skyblue")
        plt.xticks(range(X.shape[1]), [str(i) for i in features], rotation=90)
        plt.tight_layout()
        plot_filename = f"random_forest_plot_{uuid.uuid4().hex[:6]}.png"
        plot_path = os.path.join(settings.MEDIA_ROOT, plot_filename)
        plt.savefig(plot_path)
        plt.close()
        plot_url = os.path.join(settings.MEDIA_URL, plot_filename)  

        
        return render(request, 'main/random_forest.html', {
            'actual': y_test,
            'predicted': y_pred,
            'metrics': {
                'accuracy': round(accuracy, 2),
                'precision': round(precision, 2),
                'recall': round(recall, 2),
                'f1': round(f1, 2),
            },
            'download': download_link,
            'plot': plot_url,
        })
    
    return render(request, 'main/input.html', {
        'hyperparameters': {
            1: {
                'name': 'n_estimators',
                'type': 'number',
            },
        }
    })
    
def kmeans(request):
    """K-Means Clustering"""
    
    if request.method == "POST":
        dataset = request.FILES.get('dataset', None)
        n_clusters = int(request.POST.get('n_clusters'))
        
        file_extension = dataset.name.split('.')[-1]
        if file_extension == 'csv':
            df = pd.read_csv(dataset, )
        else:
            df = pd.read_excel(dataset)
        
        features = [s.replace('\n', '').replace('\r', '') for s in request.POST.getlist('features')]
                
        X = df[features]

        model = KMeans(n_clusters=n_clusters, random_state=42)
        model.fit(X)
        
        labels = model.labels_
        centroids = model.cluster_centers_
        centroids_list = [list(map(lambda x: round(x, 2), centroid)) for centroid in centroids.tolist()]
        inertia = round(model.inertia_, 2)
        silhouette = round(silhouette_score(X, labels), 2)
        
        X_data = df[features].values
        
        # ? Plotting the Clusters (Temporary)
        plot_url = None
        if (len(features) >= 2):      
            plt.scatter(X_data[:, 0], X_data[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k')
            plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=100, linewidths=2)
            plt.xlabel(X.columns[0])
            plt.ylabel(X.columns[1])
            plt.title('K-Means Clustering')
        
            plot_filename = f"kmeans_plot_{uuid.uuid4().hex[:6]}.png"
            plot_path = os.path.join(settings.MEDIA_ROOT, plot_filename)
            plt.savefig(plot_path)
            plt.close()
            plot_url = os.path.join(settings.MEDIA_URL, plot_filename)
        
        model_filename = f"kmeans_{uuid.uuid4().hex[:6]}.pkl"
        model_path = os.path.join(settings.MEDIA_ROOT, model_filename)
        
        with open(model_path, 'wb') as file:
            pickle.dump(model, file)
        
        download_link = os.path.join(settings.MEDIA_URL, model_filename)
        
        return render(request, 'main/kmeans.html', {
            'k': n_clusters,
            'X': X_data,
            'feature_count': len(features),
            'labels': labels,
            'centroids': centroids_list,
            'metrics': {
                'inertia': inertia,
                'silhouette_score': silhouette,
            },
            'plot': plot_url,
            'download': download_link,
        })
    
    return render(request, 'main/input_clustering.html', {
        'hyperparameters': {
            1: {
                'name': 'n_clusters',
                'type': 'number',
            }
        }
    })
    
    
def samples(request):
    return render(request, 'main/samples.html')

@csrf_exempt
def predict(request):
    """Open an endpoint to predict using a saved model"""
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            
            input_data = np.array(data['input']).reshape(1, -1)
            model_path = data['model_path'][1:] # Remove leading '/'
            
            with open(model_path, 'rb') as file:
                model = pickle.load(file)
            
            # Validate input data            
            expected_shape = model.n_features_in_
            if input_data.shape[1] != expected_shape:
                return JsonResponse({'error': f'Input data must have {expected_shape} features'}, status=400)
            
            predictions = model.predict(input_data)
        
            return JsonResponse({'predictions': predictions.tolist()})
        
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    return JsonResponse({'error': 'Invalid request method'}, status=400)
