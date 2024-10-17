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
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .utils import construct_line, serialize, regression_evaluation, classification_evaluation


def index(request):
    return render(request, 'main/index.html')

def classification(request):
    return render(request, 'main/algorithms.html', {
        'type': 'Classification',
        'algorithms': [
            {'name': 'K-Nearest Neighbors', 'url': 'knn',},
            {'name': 'Decision Tree', 'url': 'decision_tree',},
            {'name': 'Random Forest', 'url': 'random_forest',},
            {'name': 'Naive Bayes', 'url': 'naive_bayes',},
            {'name': 'Support Vector Machine', 'url': 'svm',},
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
        
        download_link = serialize(model, 'kmeans')
        
        return render(request, 'main/kmeans.html', {
            'k': n_clusters,
            'X': X_data,
            'features': features,
            'target': "Cluster",
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
    
# ? Other Views

def samples(request):
    datasets = [
        {
            "name": "California Housing",
            "file": "fetch_california_housing.xlsx",
            "type": "XLSX",
            "for": "Regression"
        },
        {
            "name": "California Housing",
            "file": "fetch_california_housing.csv",
            "type": "CSV",
            "for": "Regression"
        },
        {
            "name": "Numerical Data",
            "file": "numerical_data.xlsx",
            "type": "XLSX",
            "for": "Regression"
        },
        {
            "name": "Iris",
            "file": "iris.csv",
            "type": "CSV",
            "for": "Classification"
        },
        {
            "name": "Mall Customers",
            "file": "mall_customers.csv",
            "type": "CSV",
            "for": "Clustering"
        },
        {   
            "name": "Purchases",
            "file": "purchases.csv",
            "type": "CSV",
            "for": "Preprocessing"
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
