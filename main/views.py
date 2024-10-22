import csv
import json
import os
import uuid
import pickle
import numpy as np
import pandas as pd
from django.conf import settings
from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from django.core.files.storage import FileSystemStorage

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .utils import construct_line, serialize, regression_evaluation, classification_evaluation


def index(request):
    return render(request, 'main/index.html', {
        "algorithms": [
            {
                'name': 'Classification', 
                'url': 'classification',
            },
            {
                'name': 'Regression', 
                'url': 'regression',
            },
            {
                'name': 'Clustering', 
                'url': 'clustering',
            },
        ]
    })

def classification(request):
    return render(request, 'main/algorithms.html', {
        'type': 'Classification',
        'algorithms': [
            {'name': 'K-Nearest Neighbors', 'url': 'knn',},
            {'name': 'Logistic Regression', 'url': 'logistic_regression',},
            {'name': 'Naive Bayes', 'url': 'naive_bayes',},
            {'name': 'Support Vector Machine', 'url': 'svm',},
            {'name': 'Decision Tree', 'url': 'decision_tree',},
            {'name': 'Random Forest', 'url': 'random_forest',},
        ]
    })
    
def regression(request):
    return render(request, 'main/algorithms.html', {
        'type': 'Regression',
        'algorithms': [
            {'name': 'Linear Regression', 'url': 'linear_regression',},
            {'name': 'Lasso Regression', 'url': 'lasso',},
            {'name': 'Ridge Regression', 'url': 'ridge',},
            {'name': 'Decision Tree', 'url': 'decision_tree_regression',},
            {'name': 'Random Forest', 'url': 'random_forest_regression',},
        ]
    })
    
def clustering(request):
    return render(request, 'main/algorithms.html', {
        'type': 'Clustering',
        'algorithms': [
            {'name': 'K-Means', 'url': 'kmeans',},
            {'name': 'Hierarchical Clustering', 'url': 'hierarchical_clustering',},
        ]
    })

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
        
        mse, rmse, mae, r2 = regression_evaluation(y_test, y_pred)
        
        # ! Find the line equation
        intercept = model.intercept_
        coefficients = model.coef_

        equation = construct_line(intercept, coefficients, X, target)
        
        # Serialize the model and return the download link
        download_link = serialize(model, 'linear_regression')
        
        return render(request, 'main/linear_regression.html', {
            'actual': y_test,
            'predicted': y_pred_modified,
            'features': features,
            'target': target,
            'metrics': {
                'mse': round(mse, 2),
                'rmse': round(rmse, 2),
                'mae': round(mae, 2),
                'r2': round(r2, 2),
            },
            'line': equation,
            'download': download_link,
        })
        
    
    # Render the Input Form
    return render(request, 'main/input.html')
    
def lasso(request):
    """Implement Lasso Regression"""
    
    if request.method == 'POST':
        dataset = request.FILES.get('dataset', None)
        file_extension = dataset.name.split('.')[-1]
        if file_extension == 'csv':
            df = pd.read_csv(dataset, )
        else:
            df = pd.read_excel(dataset)
        
        features = [s.replace('\n', '').replace('\r', '') for s in request.POST.getlist('features')]
        target = request.POST.get('target').replace('\n', '').replace('\r', '')
        alpha = float(request.POST.get('alpha'))
                
        X, y = df[features], df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = Lasso(alpha=alpha)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_pred_modified = [round(i, 3) for i in y_pred]
        
        coeff = pd.Series(model.coef_, index=features)

        intercept = model.intercept_
        coefficients = model.coef_
        equation = construct_line(intercept, coefficients, X, target)
        
        mse, rmse, mae, r2 = regression_evaluation(y_test, y_pred)
        
        download_link = serialize(model, 'lasso')
        
        return render(request, 'main/lasso.html', {
            'coefficients': coeff,
            'actual': y_test,
            'predicted': y_pred_modified,
            'features': features,
            'target': target,
            'metrics': {
                'mse': round(mse, 2),
                'rmse': round(rmse, 2),
                'mae': round(mae, 2),
                'r2': round(r2, 2),
            },
            'download': download_link,
            'line': equation,
        })
        
    return render(request, 'main/input.html', {
        'hyperparameters': {
            1: {
                'name': 'alpha',
                'type': 'text',
            }
        }
    })
    
def ridge(request):
    """Implement Ridge Regression"""
    
    if request.method == 'POST':
        dataset = request.FILES.get('dataset', None)
        file_extension = dataset.name.split('.')[-1]
        if file_extension == 'csv':
            df = pd.read_csv(dataset, )
        else:
            df = pd.read_excel(dataset)
        
        features = [s.replace('\n', '').replace('\r', '') for s in request.POST.getlist('features')]
        target = request.POST.get('target').replace('\n', '').replace('\r', '')
        alpha = float(request.POST.get('alpha'))
                
        X, y = df[features], df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_pred_modified = [round(i, 3) for i in y_pred]
        
        coeff = pd.Series(model.coef_, index=features)

        intercept = model.intercept_
        coefficients = model.coef_
        equation = construct_line(intercept, coefficients, X, target)
        
        mse, rmse, mae, r2 = regression_evaluation(y_test, y_pred)
        
        download_link = serialize(model, 'ridge')
        
        return render(request, 'main/ridge.html', {
            'coefficients': coeff,
            'actual': y_test,
            'predicted': y_pred_modified,
            'features': features,
            'target': target,
            'metrics': {
                'mse': round(mse, 2),
                'rmse': round(rmse, 2),
                'mae': round(mae, 2),
                'r2': round(r2, 2),
            },
            'download': download_link,
            'line': equation,
        })
    
    return render(request, 'main/input.html', {
        'hyperparameters': {
            1: {
                'name': 'alpha',
                'type': 'text',
            }
        }
    })
    
def decision_tree_regression(request):
    """Decision Tree Regressor"""
    
    if request.method == 'POST':
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
        
        model = DecisionTreeRegressor(random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_pred_modified = [round(i, 3) for i in y_pred]
        
        mse, rmse, mae, r2 = regression_evaluation(y_test, y_pred)
        
        download_link = serialize(model, 'decision_tree_regression')
        
        return render(request, 'main/decision_tree_regression.html', {
            'actual': y_test,
            'predicted': y_pred_modified,
            'features': features,
            'target': target,
            'metrics': {
                'mse': round(mse, 2),
                'rmse': round(rmse, 2),
                'mae': round(mae, 2),
                'r2': round(r2, 2),
            },
            'download': download_link,
        })        
    
    return render(request, 'main/input.html')
    
def random_forest_regression(request):
    """Random Forest Regressor"""
    
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
        
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_pred_modified = [round(i, 3) for i in y_pred]
        
        mse, rmse, mae, r2 = regression_evaluation(y_test, y_pred)
        
        download_link = serialize(model, 'random_forest_regression')
        
        return render(request, 'main/random_forest_regression.html', {
            'actual': y_test,
            'predicted': y_pred_modified,
            'features': features,
            'target': target,
            'metrics': {
                'mse': round(mse, 2),
                'rmse': round(rmse, 2),
                'mae': round(mae, 2),
                'r2': round(r2, 2),
            },
            'download': download_link,
        })
    
    return render(request, 'main/input.html', {
        'hyperparameters': {
            1: {
                'name': 'n_estimators',
                'type': 'number',
            },
        }
    })
    
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
        accuracy, precision, recall, f1 = classification_evaluation(y_test, y_pred)
        
        download_link = serialize(model, 'knn')
        
        return render(request, 'main/knn.html', {
            'actual': y_test,
            'predicted': y_pred,
            'features': features,
            'target': target,
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

def logistic_regression(request):
    """Classification using Logistic Regression"""
    
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
        
        model = LogisticRegression()
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy, precision, recall, f1 = classification_evaluation(y_test, y_pred)
        
        download_link = serialize(model, 'logistic_regression')
        
        return render(request, 'main/logistic_regression.html', {
            'actual': y_test,
            'predicted': y_pred,
            'features': features,
            'target': target,
            'metrics': {
                'accuracy': round(accuracy, 2),
                'precision': round(precision, 2),
                'recall': round(recall, 2),
                'f1': round(f1, 2),
            },
            'download': download_link,
        })
    
    return render(request, 'main/input.html')

def naive_bayes(request):
    """Gaussian Naive Bayes Classifier"""
    
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
        
        model = GaussianNB()
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_pred_modified = [round(i, 3) for i in y_pred]
        accuracy, precision, recall, f1 = classification_evaluation(y_test, y_pred)
        
        download_link = serialize(model, 'naive_bayes')
        
        mean_per_class = [[round(float(j), 4) for j in i] for i in model.theta_]
        
        return render(request, 'main/naive_bayes.html', {
            'actual': y_test,
            'predicted': y_pred_modified,
            'features': features,
            'target': target,
            'results': dict(zip(model.classes_, mean_per_class)),
            'metrics': {
                'accuracy': round(accuracy, 2),
                'precision': round(precision, 2),
                'recall': round(recall, 2),
                'f1': round(f1, 2),
            },
            'download': download_link,
        })
    
    return render(request, 'main/input.html')

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
        accuracy, precision, recall, f1 = classification_evaluation(y_test, y_pred)
        
        download_link = serialize(model, 'decision_tree')
            
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
            'features': features,
            'target': target,
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
        
        accuracy, precision, recall, f1 = classification_evaluation(y_test, y_pred)
        
        download_link = serialize(model, 'random_forest')

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
            'features': features,
            'target': target,
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
       
def svm(request):
    """Support Vector Machine"""
    
    if request.method == "POST":
        dataset = request.FILES.get('dataset', None)
        file_extension = dataset.name.split('.')[-1]
        if file_extension == 'csv':
            df = pd.read_csv(dataset, )
        else:
            df = pd.read_excel(dataset)
        
        features = [s.replace('\n', '').replace('\r', '') for s in request.POST.getlist('features')]
        target = request.POST.get('target').replace('\n', '').replace('\r', '')
        kernel = request.POST.get('kernel')
        C = float(request.POST.get('C'))
        
        X, y = df[features], df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = SVC(kernel=kernel, C=C, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy, precision, recall, f1 = classification_evaluation(y_test, y_pred)
        
        download_link = serialize(model, 'svm')
        
        return render(request, 'main/svm.html', {
            'actual': y_test,
            'predicted': y_pred,
            'features': features,
            'target': target,
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
                'field': 'select',
                'name': 'kernel',
                'type': 'text',
                'options': ['linear', 'poly', 'rbf', 'sigmoid'],
            },
            2: {
                'name': 'C',
                'type': 'text',
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
        
        download_link = serialize(model, 'kmeans')
        
        return render(request, 'main/kmeans.html', {
            'k': n_clusters,
            'X': X_data,
            'features': features,
            'target': "Cluster", # For prediction
            'feature_count': len(features),
            'labels': labels,
            'centroids': centroids_list,
            'metrics': {
                'inertia': inertia,
                'silhouette_score': silhouette,
            },
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
    
def hierarchical_clustering(request):
    """Agglomerative Hierarchical Clustering"""
    
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

        model = AgglomerativeClustering(n_clusters=n_clusters)
        labels = model.fit_predict(X)
        centroids = np.array([X[model.labels_ == i].mean(axis=0) for i in np.unique(model.labels_)])
        centroids_list = [list(map(lambda x: round(x, 2), centroid)) for centroid in centroids.tolist()]
        silhouette = round(silhouette_score(X, labels), 2)
        
        X_data = df[features].values
        
        # ? Plotting the Dendrogram
        plot_url = None
        if (len(features) >= 2):      
            linked = linkage(X_data, 'ward')
            plt.figure(figsize=(10, 7))
            dendrogram(linked, orientation='top', labels=df.index, distance_sort='descending', show_leaf_counts=True)
            plt.title('Dendrogram')
        
            plot_filename = f"hierarchical_clustering_plot_{uuid.uuid4().hex[:6]}.png"
            plot_path = os.path.join(settings.MEDIA_ROOT, plot_filename)
            plt.savefig(plot_path)
            plt.close()
            plot_url = os.path.join(settings.MEDIA_URL, plot_filename)
        
        download_link = serialize(model, 'hierarchical_clustering')
        
        return render(request, 'main/hierarchical_clustering.html', {
            'k': n_clusters,
            'X': X_data,
            'features': features,
            'target': "Cluster",
            'feature_count': len(features),
            'labels': labels,
            'centroids': centroids_list,
            'metrics': {
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
    
# ? Other Views

def samples(request):
    datasets = [
        {
            "name": "California Housing",
            "file": "fetch_california_housing.xlsx",
            "type": "XLSX",
            "note": "For Regression (Large)"
        },
        {
            "name": "California Housing",
            "file": "fetch_california_housing.csv",
            "type": "CSV",
            "note": "For Regression (Large)"
        },
        {
            "name": "Numerical Data",
            "file": "numerical_data.xlsx",
            "type": "XLSX",
            "note": "For Regression (Large)"
        },
        {
            "name": "Iris",
            "file": "iris.csv",
            "type": "CSV",
            "note": "For Classification (Small)"
        },
        {
            "name": "Mall Customers",
            "file": "mall_customers.csv",
            "type": "CSV",
            "note": "For Clustering (Small)"
        },
        {   
            "name": "Countries and Purchases",
            "file": "purchases.csv",
            "type": "CSV",
            "note": "Uncleaned (Small)"
        },
        {
            "name": "Pima Indians Diabetes",
            "file": "diabetes.csv",
            "type": "CSV",
            "note": "For Classification (Binary)"
        },
        {
            "name": "Big Mart Sales",
            "file": "big_mart_sales.csv",
            "type": "CSV",
            "note": "Uncleaned (Large)"
        }
    ]
    return render(request, 'main/samples.html', {
        'datasets': datasets,
    })

# ? API Endpoints
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
            print(predictions)
        
            return JsonResponse({'predictions': [round(i, 4) for i in predictions.tolist()]})
        
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    return JsonResponse({'error': 'Invalid request method'}, status=400)

# ? Preprocessing

def preprocessing(request):
    context = {}

    if request.method == 'POST' and request.FILES.get('file'):
        uploaded_file = request.FILES['file']
        try:
            # Read the uploaded file into a DataFrame
            data = pd.read_csv(uploaded_file)

            # Store the initial dataset in the session
            request.session['updated_data'] = data.to_dict()

            # Prepare the data preview for rendering
            context['data_preview'] = data.to_html(classes='table table-bordered table-hover', index=False)
            context['headers'] = data.columns.tolist()  # Store headers for use in the template

        except Exception as e:
            context['error'] = f"Error processing data: {e}"

    return render(request, 'main/preprocessing.html', context)

def fill_missing_values(request):
    if request.method == 'POST':
        # Load the updated data from session
        data_dict = request.session.get('updated_data')
        if not data_dict:
            return JsonResponse({'error': 'No data available'}, status=400)

        data = pd.DataFrame.from_dict(data_dict)
        
        databody = json.loads(request.body)

        missing_value_strategy = databody.get('strategy')
        selected_columns = databody.get('columns')
        
        
        # Apply missing value handling logic
        if missing_value_strategy and selected_columns:
            
            for col in selected_columns:
                if data[col].dtype != 'object':  # Ensure column is numerical
                    if missing_value_strategy == 'mean':
                        data[col].fillna(data[col].mean(), inplace=True)
                    elif missing_value_strategy == 'median':
                        data[col].fillna(data[col].median(), inplace=True)
                    elif missing_value_strategy == 'drop':
                        data.dropna(subset=selected_columns, inplace=True)

        # Update session with new data
        request.session['updated_data'] = data.to_dict()
        return JsonResponse({'data_preview': data.to_html(classes='table table-bordered', index=False)})
    
def encoding(request):
    if request.method == 'POST':
        # Load the updated data from session
        data_dict = request.session.get('updated_data')
        if not data_dict:
            return JsonResponse({'error': 'No data available'}, status=400)

        data = pd.DataFrame.from_dict(data_dict)
        databody = json.loads(request.body)

        encoding_strategy = databody.get('strategy')
        encoding_columns = databody.get('columns')
        # Apply missing value handling logic
        if encoding_strategy == 'onehot' and encoding_columns:
                data = pd.get_dummies(data, columns=encoding_columns)
        elif encoding_strategy == 'label' and encoding_columns:
            le = LabelEncoder()
            for col in encoding_columns:
                if data[col].dtype == 'object':  # Ensure column is categorical
                    data[col] = le.fit_transform(data[col])


        # Update session with new data
        request.session['updated_data'] = data.to_dict()
        return JsonResponse({'data_preview': data.to_html(classes='table table-bordered', index=False)})
  
def scaling(request):
    if request.method == 'POST':
        # Load the updated data from session
        data_dict = request.session.get('updated_data')
        if not data_dict:
            return JsonResponse({'error': 'No data available'}, status=400)

        data = pd.DataFrame.from_dict(data_dict)
        databody = json.loads(request.body)

        scaling_strategy = databody.get('strategy')

        if scaling_strategy == 'standard':
            scaler = StandardScaler()
            data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
        elif scaling_strategy == 'normalize':
            scaler = MinMaxScaler()
            data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

        # Update session with new data
        request.session['updated_data'] = data.to_dict()
        return JsonResponse({'data_preview': data.to_html(classes='table table-bordered', index=False)})
 
def download_csv(request):
    data_dict=request.session.get('updated_data',None)
    if data_dict:
        # Convert the dictionary back to a DataFrame
        data = pd.DataFrame.from_dict(data_dict)

        # Create the HttpResponse object with the appropriate CSV header
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="updated_data.csv"'

        # Create a CSV writer
        writer = csv.writer(response)

        # Write the headers (columns) of your CSV file
        writer.writerow(data.columns)

        # Write the data rows
        for index, row in data.iterrows():
            writer.writerow(row)

        return response
    else:
        # Handle case where session data is not available
        return HttpResponse("No data available", status=400)

