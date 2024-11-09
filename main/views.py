import csv
import json
import pickle
import numpy as np
import pandas as pd
from django.conf import settings
from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse, HttpResponse, Http404
from django.views.decorators.csrf import csrf_exempt
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

from .utils import get_input, construct_line, format_predictions, regression_evaluation, classification_evaluation
from .utils import plot_feature_importances, plot_decision_tree, plot_dendrogram, plot_kmeans_clusters
from .models import MLModel, DataFile


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
    if request.method == 'POST':
        file_id = request.session.get('file', None)
        file_model = get_object_or_404(DataFile, file_id=file_id)
        df = file_model.load_file()

        features, target, test_size, fit_intercept = get_input(request.POST, 'fit_intercept')
        fit_intercept = fit_intercept == 'true'
        
        X, y = df[features], df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        model = LinearRegression(fit_intercept=fit_intercept)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        metrics = regression_evaluation(y_test, y_pred)
        intercept = model.intercept_
        coefficients = model.coef_
        equation = construct_line(intercept, coefficients, X, target)

        ml_model = MLModel()
        ml_model.save_model(model)
        request.session['model'] = str(ml_model.model_id)

        return render(request, 'main/linear_regression.html', {
            'actual': y_test[:100],
            'predicted': format_predictions(y_pred),
            'features': features,
            'target': target,
            'metrics': metrics,
            'line': equation,
        })

    return render(request, 'main/input.html', {
        'optional_parameters': [
            {'field': 'checkbox', 'name': 'fit_intercept', 'type': 'checkbox', 'default': 'true'},
        ]
    })
    
def lasso(request):
    """Implement Lasso Regression"""
    
    if request.method == 'POST':
        file_id = request.session.get('file', None)
        file_model = get_object_or_404(DataFile, file_id=file_id)
        df = file_model.load_file()
        
        features, target, test_size, alpha, max_iter, tol = get_input(request.POST, 'alpha', ('max_iter', 1000), ('tol', 1e-4))
        alpha, max_iter, tol = float(alpha), int(max_iter), float(tol)
        
        X, y = df[features], df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        model = Lasso(alpha=alpha, max_iter=max_iter, tol=tol)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        coeff = pd.Series(model.coef_, index=features)
        intercept = model.intercept_
        equation = construct_line(intercept, model.coef_, X, target)
        metrics = regression_evaluation(y_test, y_pred)
        
        ml_model = MLModel()
        ml_model.save_model(model)
        request.session['model'] = str(ml_model.model_id)
        
        return render(request, 'main/lasso.html', {
            'coefficients': coeff,
            'actual': y_test[:100],
            'predicted': format_predictions(y_pred),
            'features': features,
            'target': target,
            'metrics': metrics,
            'line': equation,
        })
    
    return render(request, 'main/input.html', {
        'hyperparameters': {
            1: {'name': 'alpha', 'type': 'text', 'default': 1.0},
        },
        'optional_parameters': [
            {'name': 'max_iter', 'type': 'number', 'default': 1000},
            {'name': 'tol', 'type': 'text', 'default': 1e-4},
        ]
    })
    
def ridge(request):
    """Implement Ridge Regression"""

    if request.method == 'POST':
        file_id = request.session.get('file', None)
        file_model = get_object_or_404(DataFile, file_id=file_id)
        df = file_model.load_file()

        features, target, test_size, alpha, max_iter, tol = get_input(request.POST, 'alpha', ('max_iter', 1000), ('tol', 1e-4))
        alpha, max_iter, tol = float(alpha), int(max_iter), float(tol)

        X, y = df[features], df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        model = Ridge(alpha=alpha, max_iter=max_iter, tol=tol)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        coeff = pd.Series(model.coef_, index=features)
        intercept = model.intercept_
        equation = construct_line(intercept, model.coef_, X, target)
        metrics = regression_evaluation(y_test, y_pred)

        ml_model = MLModel()
        ml_model.save_model(model)
        request.session['model'] = str(ml_model.model_id)

        return render(request, 'main/ridge.html', {
            'coefficients': coeff,
            'actual': y_test[:100],
            'predicted': format_predictions(y_pred),
            'features': features,
            'target': target,
            'metrics': metrics,
            'line': equation,
        })

    return render(request, 'main/input.html', {
        'hyperparameters': {
            1: {'name': 'alpha', 'type': 'text', 'default': 1.0},
        },
        'optional_parameters': [
            {'name': 'max_iter', 'type': 'number', 'default': 1000},
            {'name': 'tol', 'type': 'text', 'default': 1e-4},
        ]
    })
    
def decision_tree_regression(request):
    """Decision Tree Regressor"""

    if request.method == 'POST':
        file_id = request.session.get('file', None)
        file_model = get_object_or_404(DataFile, file_id=file_id)
        df = file_model.load_file()

        features, target, test_size, max_depth, min_samples_split = get_input(request.POST, ('max_depth', None), ('min_samples_split', 2))
        min_samples_split = int(min_samples_split)

        X, y = df[features], df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        if not max_depth:
            model = DecisionTreeRegressor(min_samples_split=min_samples_split, random_state=42)
        else:
            model = DecisionTreeRegressor(max_depth=int(max_depth), min_samples_split=min_samples_split, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        metrics = regression_evaluation(y_test, y_pred)

        ml_model = MLModel()
        ml_model.save_model(model)
        request.session['model'] = str(ml_model.model_id)

        return render(request, 'main/decision_tree_regression.html', {
            'actual': y_test[:100],
            'predicted': format_predictions(y_pred),
            'features': features,
            'target': target,
            'metrics': metrics,
        })

    return render(request, 'main/input.html', {
        'optional_parameters': [
            {'field': 'input', 'name': 'max_depth', 'type': 'number'},
            {'field': 'input', 'name': 'min_samples_split', 'type': 'number', 'default': 2},
        ]
    })
    
def random_forest_regression(request):
    """Random Forest Regression"""

    if request.method == 'POST':
        file_id = request.session.get('file', None)
        file_model = get_object_or_404(DataFile, file_id=file_id)
        df = file_model.load_file()

        features, target, test_size, n_estimators, max_depth, min_samples_split = get_input(request.POST, 'n_estimators', ('max_depth', None), ('min_samples_split', 2))
        n_estimators, min_samples_split = int(n_estimators), int(min_samples_split)

        X, y = df[features], df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        if not max_depth:
            model = RandomForestRegressor(n_estimators=n_estimators, min_samples_split=min_samples_split, random_state=42)
        else:
            model = RandomForestRegressor(n_estimators=n_estimators, max_depth=int(max_depth), min_samples_split=min_samples_split, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        metrics = regression_evaluation(y_test, y_pred)

        ml_model = MLModel()
        ml_model.save_model(model)
        request.session['model'] = str(ml_model.model_id)

        return render(request, 'main/random_forest_regression.html', {
            'actual': y_test[:100],
            'predicted': format_predictions(y_pred),
            'features': features,
            'target': target,
            'metrics': metrics,
        })

    return render(request, 'main/input.html', {
        'hyperparameters': {
            1: {'name': 'n_estimators', 'type': 'number'},
        },
        'optional_parameters': [
            {'field': 'input', 'name': 'max_depth', 'type': 'number'},
            {'field': 'input', 'name': 'min_samples_split', 'type': 'number', 'default': 2},
        ]
    })
    
def knn(request):
    """Build KNN model and evaluate it"""
    
    if request.method == "POST":
        file_id = request.session.get('file', None)
        file_model = get_object_or_404(DataFile, file_id=file_id)
        df = file_model.load_file()
        
        features, target, test_size, n_neighbors, weights, algorithm, metric, p = get_input(request.POST, 'n_neighbors', 'weights', 'algorithm', 'metric', 'p')
        n_neighbors, p = int(n_neighbors), int(p)
                
        X, y = df[features], df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm, metric=metric, p=p)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        metrics = classification_evaluation(y_test, y_pred)
        
        ml_model = MLModel()
        ml_model.save_model(model)
        request.session['model'] = str(ml_model.model_id)        
        
        return render(request, 'main/knn.html', {
            'actual': y_test[:100],
            'predicted': y_pred[:100],
            'features': features,
            'target': target,
            'metrics': metrics,
        })
    
    return render(request, 'main/input.html', {
        'hyperparameters': {
            1: {'name': 'n_neighbors', 'type': 'number'},
        },
        'optional_parameters': [
            {'field': 'select', 'name': 'weights', 'type': 'text', 'options': ['uniform', 'distance'], 'default': 'uniform'},
            {'field': 'select', 'name': 'algorithm', 'type': 'text', 'options': ['auto', 'ball_tree', 'kd_tree', 'brute'], 'default': 'auto'},
            {'field': 'select', 'name': 'metric', 'type': 'text', 'options': ['minkowski', 'euclidean', 'manhattan', 'cosine', 'hamming'], 'default': 'minkowski'},
            {'field': 'select', 'name': 'p', 'type': 'number', 'options': [1, 2], 'default': 2}
        ]
    })

def logistic_regression(request):
    """Classification using Logistic Regression"""
    
    if request.method == "POST":
        file_id = request.session.get('file', None)
        file_model = get_object_or_404(DataFile, file_id=file_id)
        df = file_model.load_file()
        
        features, target, test_size, solver, penalty, C = get_input(request.POST, 'solver', 'penalty', 'C')
        if penalty == 'none': penalty = None
        C = float(C)
                
        X, y = df[features], df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        model = LogisticRegression(solver=solver, penalty=penalty, C=C)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        metrics = classification_evaluation(y_test, y_pred)
        
        ml_model = MLModel()
        ml_model.save_model(model)
        request.session['model'] = str(ml_model.model_id)        
        
        return render(request, 'main/logistic_regression.html', {
            'actual': y_test[:100],
            'predicted': y_pred[:100],
            'features': features,
            'target': target,
            'metrics': metrics,
        })
    
    return render(request, 'main/input.html', {
        'optional_parameters': [
            {'field': 'select', 'name': 'solver', 'type': 'text', 'options': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'], 'default': 'lbfgs'},
            {'field': 'select', 'name': 'penalty', 'type': 'text', 'options': ['l2', 'None', 'elasticnet', 'l1'], 'default': 'l2'},
            {'field': 'input', 'name': 'C', 'type': 'number', 'default': 1.0},
        ]
    })

def naive_bayes(request):
    """Gaussian Naive Bayes Classifier"""
    
    if request.method == "POST":
        file_id = request.session.get('file', None)
        file_model = get_object_or_404(DataFile, file_id=file_id)
        df = file_model.load_file()
        
        features, target, test_size, var_smoothing = get_input(request.POST, 'var_smoothing')
        var_smoothing = float(var_smoothing)
                
        X, y = df[features], df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        model = GaussianNB(var_smoothing=var_smoothing)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_pred_modified = [round(i, 3) for i in y_pred]
        metrics = classification_evaluation(y_test, y_pred)
        
        ml_model = MLModel()
        ml_model.save_model(model)
        request.session['model'] = str(ml_model.model_id)        
        
        mean_per_class = [[round(float(j), 4) for j in i] for i in model.theta_]
        
        return render(request, 'main/naive_bayes.html', {
            'actual': y_test[:100],
            'predicted': y_pred_modified[:100],
            'features': features,
            'target': target,
            'results': dict(zip(model.classes_, mean_per_class)),
            'metrics': metrics,
        })
    
    return render(request, 'main/input.html', {
        'optional_parameters': [
            {'field': 'input', 'name': 'var_smoothing', 'type': 'text', 'default': 1e-9},
        ]
    })

def decision_tree(request):
    """Decision Tree Classifier"""
    
    if request.method == "POST":
        file_id = request.session.get('file', None)
        file_model = get_object_or_404(DataFile, file_id=file_id)
        df = file_model.load_file()
        
        features, target, test_size, max_depth, min_samples_split = get_input(request.POST, ('max_depth', None), ('min_samples_split', 2))
        min_samples_split = int(min_samples_split)
                
        X, y = df[features], df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        if not max_depth:
            model = DecisionTreeClassifier(min_samples_split=min_samples_split, random_state=42)
        else:
            model = DecisionTreeClassifier(max_depth=int(max_depth), min_samples_split=min_samples_split, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        metrics = classification_evaluation(y_test, y_pred)
        
        ml_model = MLModel()
        ml_model.save_model(model)
        request.session['model'] = str(ml_model.model_id)        
            
        graph_json = plot_decision_tree(model, features)
        
        return render(request, 'main/decision_tree.html', {
            'actual': y_test[:100],
            'predicted': y_pred[:100],
            'features': features,
            'target': target,
            'metrics': metrics,
            'tree': graph_json,
        })
    
    return render(request, 'main/input.html', {
        'optional_parameters': [
            {'field': 'input', 'name': 'max_depth', 'type': 'number'},
            {'field': 'input', 'name': 'min_samples_split', 'type': 'number', 'default': 2},
        ]
    })

def random_forest(request):
    """Random Forest Classifier"""
    
    if request.method == "POST":
        file_id = request.session.get('file', None)
        file_model = get_object_or_404(DataFile, file_id=file_id)
        df = file_model.load_file()

        features, target, test_size, n_estimators, max_depth, min_samples_split = get_input(request.POST, 'n_estimators', ('max_depth', None), ('min_samples_split', 2))
        n_estimators, min_samples_split = int(n_estimators), int(min_samples_split)
                
        X, y = df[features], df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        if not max_depth:
            model = RandomForestClassifier(n_estimators=n_estimators, min_samples_split=min_samples_split, random_state=42)
        else:
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=int(max_depth), min_samples_split=min_samples_split, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        metrics = classification_evaluation(y_test, y_pred)
        
        ml_model = MLModel()
        ml_model.save_model(model)
        request.session['model'] = str(ml_model.model_id)        

        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        graph_json = plot_feature_importances(features, importances, indices)
        
        return render(request, 'main/random_forest.html', {
            'actual': y_test[:100],
            'predicted': y_pred[:100],
            'features': features,
            'target': target,
            'metrics': metrics,
            'graph': graph_json,
        })
    
    return render(request, 'main/input.html', {
        'hyperparameters': {
            1: {'name': 'n_estimators', 'type': 'number'},
        },
        'optional_parameters': [
            {'field': 'input', 'name': 'max_depth', 'type': 'number'},
            {'field': 'input', 'name': 'min_samples_split', 'type': 'number', 'default': 2},
        ]
    })

def svm(request):
    """Build SVM model and evaluate it"""

    if request.method == "POST":
        file_id = request.session.get('file', None)
        file_model = get_object_or_404(DataFile, file_id=file_id)
        df = file_model.load_file()

        features, target, test_size, kernel, C, gamma, degree = get_input(request.POST, 'kernel', 'C', 'gamma', 'degree')
        C, degree = float(C), int(degree)

        X, y = df[features], df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        model = SVC(kernel=kernel, C=C, gamma=gamma, degree=degree, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        metrics = classification_evaluation(y_test, y_pred)

        ml_model = MLModel()
        ml_model.save_model(model)
        request.session['model'] = str(ml_model.model_id)

        return render(request, 'main/svm.html', {
            'actual': y_test[:100],
            'predicted': y_pred[:100],
            'features': features,
            'target': target,
            'metrics': metrics,
        })

    return render(request, 'main/input.html', {
        'hyperparameters': {
            1: {'field': 'select', 'name': 'kernel', 'type': 'text', 'options': ['linear', 'poly', 'rbf', 'sigmoid'], 'default': 'rbf'},
            2: {'name': 'C', 'type': 'text', 'default': 1.0},
        },
        'optional_parameters': [
            {'name': 'gamma', 'type': 'select', 'field': 'select', 'options': ['scale', 'auto'], 'default': 'scale'},
            {'name': 'degree', 'type': 'number', 'default': 3}
        ]
    })

def kmeans(request):
    """K-Means Clustering"""
    
    if request.method == "POST":
        file_id = request.session.get('file', None)
        file_model = get_object_or_404(DataFile, file_id=file_id)
        df = file_model.load_file()
        
        features = request.POST.getlist('features')                
        n_clusters = int(request.POST.get('n_clusters'))
        
        X = df[features]

        model = KMeans(n_clusters=n_clusters, random_state=42)
        model.fit(X)
        
        labels = model.labels_
        centroids = model.cluster_centers_
        centroids_list = [list(map(lambda x: round(x, 2), centroid)) for centroid in centroids.tolist()]
        inertia = round(model.inertia_, 2)
        silhouette = round(silhouette_score(X, labels), 2)
        
        X_data = df[features].values
        
        ml_model = MLModel()
        ml_model.save_model(model)
        request.session['model'] = str(ml_model.model_id)        

        plot_json = None
        if (len(features) >= 2):
            plot_json = plot_kmeans_clusters(X_data, labels, centroids, features, 0, 1)
        
        return render(request, 'main/kmeans.html', {
            'k': n_clusters,
            'X': X_data[:100],
            'features': features,
            'target': "Cluster", # For prediction
            'feature_count': len(features),
            'labels': labels[:100],
            'centroids': centroids_list,
            'metrics': {
                'inertia': inertia,
                'silhouette_score': silhouette,
            },
            'plot': plot_json,
        })
    
    return render(request, 'main/input_clustering.html', {
        'hyperparameters': {
            1: {'name': 'n_clusters', 'type': 'number'}
        }
    })
    
def hierarchical_clustering(request):
    """Agglomerative Hierarchical Clustering"""
    
    if request.method == "POST":
        file_id = request.session.get('file', None)
        file_model = get_object_or_404(DataFile, file_id=file_id)
        df = file_model.load_file()
        
        features = request.POST.getlist('features')                
        n_clusters = int(request.POST.get('n_clusters'))
        linkage_method = request.POST.get('linkage_method', 'ward')
        
        X = df[features]

        model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
        labels = model.fit_predict(X)
        
        centroids = np.array([X[model.labels_ == i].mean(axis=0) for i in np.unique(model.labels_)])
        centroids_list = [list(map(lambda x: round(x, 2), centroid)) for centroid in centroids.tolist()]
        silhouette = round(silhouette_score(X, labels), 2)
        
        X_data = df[features].values
        
        # ? Plotting the Dendrogram
        plot_json = None
        if (len(features) >= 2):
            linked = linkage(X_data, 'ward') # Using Ward linkage method
            plot_json = plot_dendrogram(linked, df.index)
        
        ml_model = MLModel()
        ml_model.save_model(model)
        request.session['model'] = str(ml_model.model_id)        
        
        return render(request, 'main/hierarchical_clustering.html', {
            'k': n_clusters,
            'X': X_data[:100],
            'features': features,
            'target': "Cluster",
            'feature_count': len(features),
            'labels': labels[:100],
            'centroids': centroids_list,
            'metrics': {
                'silhouette_score': silhouette,
            },
            'dendrogram': plot_json,
        })
    
    return render(request, 'main/input_clustering.html', {
        'hyperparameters': {
            1: {'name': 'n_clusters', 'type': 'number'},
        },
        'optional_parameters': [
            {'field': 'select', 'name': 'linkage_method', 'type': 'text', 'options': ['ward', 'complete', 'average', 'single'], 'default': 'ward'},
        ]
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

def download_model(request):
    """Download the trained model stored in the database"""
    # Retrieve the model ID from the session
    model_id = request.session.get('model')
    if not model_id:
        raise Http404("Model ID not found in session.")

    # Retrieve the model from the database using the model ID
    ml_model = get_object_or_404(MLModel, model_id=model_id)
    
    # Create an HTTP response with the model data as a downloadable file
    response = HttpResponse(ml_model.model_data, content_type='application/octet-stream')
    response['Content-Disposition'] = f'attachment; filename="model-{model_id[:5]}.pkl"'
    
    return response

def test(request):
    return render(request, 'main/temp.html')

# ? API Endpoints
@csrf_exempt
def predict(request):
    """Open an endpoint to predict using a saved model"""
    if request.method == "POST":
        try:            
            # Load the model
            model_id = request.session.get('model')
            if not model_id:
                return JsonResponse({'error': 'No model available'}, status=400)
            
            ml_model = get_object_or_404(MLModel, model_id=model_id)
            model = ml_model.load_model()
                        
            # Validate input data            
            data = json.loads(request.body)
            input_data = np.array(data['input']).reshape(1, -1)
            expected_shape = model.n_features_in_
            if input_data.shape[1] != expected_shape:
                return JsonResponse({'error': f'Input data must have {expected_shape} features'}, status=400)
            
            predictions = model.predict(input_data)        
            return JsonResponse({'predictions': [round(i, 4) for i in predictions.tolist()]})
        
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    return JsonResponse({'error': 'Invalid request method'}, status=400)

def save_file(request):
    """Save the uploaded file to the database as JSON"""
    if request.method == 'POST' and request.FILES.get('file'):
        file = request.FILES['file']
        
        # Check file size and format
        if file.size > 2 * 1024 * 1024: # 2MB
            return JsonResponse({'error': 'File size is too large. Max file size is 2MB'}, status=400)        
        # Use pandas to read the file based on its extension
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.name.endswith('.xls') or file.name.endswith('.xlsx'):
            df = pd.read_excel(file)
        else:
            return JsonResponse({'error': 'Invalid file format. Only CSV and Excel files are allowed'}, status=400)

        
        # Store the file name and file as JSON in the db, store the id in the session
        file_model = DataFile()
        file_model.save_file(file.name, df)    
        request.session['file'] = str(file_model.file_id)

        return JsonResponse({'message': 'File uploaded successfully!'})
    
    return JsonResponse({'error': 'Invalid request method'}, status=400)

def get_file(request):
    """Return the file content stored in the session"""
    if request.method == 'POST':
        file_model = get_object_or_404(DataFile, file_id=request.session.get('file'))
        filename, df = file_model.filename, file_model.load_file()
    
        if not df.empty:
            columns = df.columns.tolist()            
            correlation_matrix = df.corr()

            return JsonResponse({
                'filename': filename,
                'file': df.to_dict(),
                'columns': columns,
                'correlation_matrix': correlation_matrix.to_dict(),
            })
        return JsonResponse({'Error': 'No file available'}, status=400)
    return JsonResponse({'error': 'Invalid request method'}, status=400)

# ? Preprocessing

def preprocessing(request):
    """Store the uploaded file and display the data preview"""
    
    if request.method == 'POST' and request.FILES.get('file'):
        uploaded_file = request.FILES['file']
        try:
            # Read the uploaded file into a DataFrame
            data = pd.read_csv(uploaded_file)
            
            # Store the initial dataset in the database
            file_model = DataFile()
            file_model.save_file(uploaded_file.name, data)
            request.session['file'] = str(file_model.file_id)

            # Get the columns with missing values and non-numerical columns
            null_columns = data.columns[data.isnull().any()]
            non_numerical_cols = data.select_dtypes(include=['object', 'category']).columns

            

            # Prepare the data preview for rendering
            json_data = data.head(20).to_json(orient='records')
            headers = data.columns.tolist()
            null_columns = null_columns.tolist()
            non_numerical_cols = non_numerical_cols.tolist() # Cols with categorical values
            
            return JsonResponse({
                'json_data': json_data,
                'headers': headers,
                'null_columns': null_columns,
                'non_numerical_cols':non_numerical_cols
            })
            
        except Exception as e:
            return JsonResponse({'error': f"Error processing data: {e}"})

    return render(request, 'main/preprocessing.html')

def fill_missing_values(request):
    """Replace missing values with mean / median or drop the rows"""
    
    if request.method == 'POST':
        # Load the file
        file_id = request.session.get('file', None)
        file_model = get_object_or_404(DataFile, file_id=file_id)
        data = file_model.load_file()

        if data.empty:
            return JsonResponse({'error': 'No data available'}, status=400)
        
        databody = json.loads(request.body)

        missing_value_strategy = databody.get('strategy')
        selected_columns = databody.get('columns')
        
        if not missing_value_strategy or not selected_columns:
            return JsonResponse({'error': 'Invalid input, strategy and columns are required'}, status=400)
        
        else:
            

    
            # Handle "drop" strategy separately
            if missing_value_strategy == 'drop':
                data.dropna(subset=selected_columns, inplace=True)
            else:
                # Loop through each column in selected_columns
                for col in selected_columns:
                    # Determine imputer strategy based on column type and missing_value_strategy
                    if data[col].dtype != 'object' :
                        imputer = SimpleImputer(missing_values=np.nan,strategy=missing_value_strategy)
                    elif missing_value_strategy == 'most_frequent':                        
                        imputer = SimpleImputer(missing_values=None,strategy='most_frequent')
                    else:
                        raise ValueError(f"Unsupported missing_value_strategy '{missing_value_strategy}' for column '{col}'")

                    # Apply the imputer to the column
                    data[[col]] = imputer.fit_transform(data[[col]])
   
        # Save the updated data back to the database
        file_model.save_file(file_model.filename, data)
        request.session['file'] = str(file_model.file_id)

        # Return the updated data preview
        null_columns = data.columns[data.isnull().any()]
        non_numerical_cols = data.select_dtypes(include=['object', 'category']).columns

        json_data = data.head(20).to_json(orient='records')
        headers = data.columns.tolist()
        null_columns = null_columns.tolist()
        non_numerical_cols = non_numerical_cols.tolist() 
        return JsonResponse({
            'json_data': json_data,
            'headers': headers,
            'null_columns': null_columns,
            'non_numerical_cols':non_numerical_cols
        })

def encoding(request):
    """
    Encoding categorical columns into numerical values
    One-Hot Encoding: Convert each category value into a new column and assigns a 1 or 0 (True/False) value to the column.
    Label Encoding: Convert each category value into a unique integer value.
    """
    if request.method == 'POST':
        file_id = request.session.get('file', None)
        file_model = get_object_or_404(DataFile, file_id=file_id)
        data = file_model.load_file()
        if data.empty:
            return JsonResponse({'error': 'No data available'}, status=400)

        # Get the encoding strategy and columns to encode
        databody = json.loads(request.body)
        encoding_strategy = databody.get('strategy')
        encoding_columns = databody.get('columns')

        if not encoding_strategy or not encoding_columns:
            return JsonResponse({'error': 'Invalid input, strategy and columns are required'}, status=400)

        # Apply missing value handling logic
        if encoding_strategy == 'onehot' and encoding_columns:
            data = pd.get_dummies(data, columns=encoding_columns,dtype=int)
        elif encoding_strategy == 'label' and encoding_columns:
            le = LabelEncoder()
            for col in encoding_columns:
                if data[col].dtype == 'object':  # Ensure column is categorical
                    data[col] = le.fit_transform(data[col])


        # Update data in the database
        file_model.save_file(file_model.filename, data)
        request.session['file'] = str(file_model.file_id)
        
        # Return the updated data preview
        null_columns = data.columns[data.isnull().any()]
        non_numerical_cols = data.select_dtypes(include=['object', 'category']).columns

        json_data = data.head(20).to_json(orient='records')
        headers = data.columns.tolist()
        null_columns = null_columns.tolist()
        non_numerical_cols = non_numerical_cols.tolist()  #columns with categorical values
        return JsonResponse({
            'json_data': json_data,
            'headers': headers,
            'null_columns': null_columns,
            'non_numerical_cols':non_numerical_cols
        })
  
def scaling(request):
    """
    Perform Normalization or Standardization on the data
    Min-Max Scaling: Scale the data between 0 and 1
    Standard Scaling: Scale the data to have a mean of 0 and a standard deviation of 1
    """
    if request.method == 'POST':
        # Load the updated data from session
        # data_dict = request.session.get('updated_data')
        file_id = request.session.get('file', None)
        file_model = get_object_or_404(DataFile, file_id=file_id)
        data = file_model.load_file()
        if data.empty:
            return JsonResponse({'error': 'No data available'}, status=400)

        # Parse the request body
        databody = json.loads(request.body)

        # Get the scaling strategy and columns to scale
        scaling_strategy = databody.get('strategy')
        scaling_columns = databody.get('columns')

        if not scaling_strategy or not scaling_columns:
            return JsonResponse({'error': 'Invalid input, strategy and columns are required'}, status=400)

        # Ensure the columns exist in the data
        if not all(col in data.columns for col in scaling_columns):
            return JsonResponse({'error': 'One or more columns do not exist in the data'}, status=400)

        # Apply the appropriate scaling strategy
        if scaling_strategy == 'normalize':
            scaler = MinMaxScaler()
        elif scaling_strategy == 'standard':
            scaler = StandardScaler()
        else:
            return JsonResponse({'error': 'Invalid scaling strategy'}, status=400)

        try:
            # Perform scaling on the specified columns
            data[scaling_columns] = scaler.fit_transform(data[scaling_columns])
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

        # Store the scaled data back into the session
        file_model.save_file(file_model.filename, data)
        request.session['file'] = str(file_model.file_id)

        # Return the updated data preview
        null_columns = data.columns[data.isnull().any()]
        non_numerical_cols = data.select_dtypes(include=['object', 'category']).columns

        json_data = data.head(20).to_json(orient='records')
        headers = data.columns.tolist()
        null_columns = null_columns.tolist()
        non_numerical_cols=non_numerical_cols.tolist()  #columns with categorical values
        return JsonResponse({
                'json_data': json_data,
                'headers': headers,
                'null_columns': null_columns,
                'non_numerical_cols':non_numerical_cols
            })

    return JsonResponse({'error': 'Invalid request method'}, status=405)

def download_csv(request):
    """Download the updated data"""
    # data_dict=request.session.get('updated_data',None)
    file_id = request.session.get('file', None)
    file_model = get_object_or_404(DataFile, file_id=file_id)
    data = file_model.load_file()
    
    if not data.empty:
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

def data_details(request):
    """Display data statistics such as missing values, mean, median, etc."""
   
    # Get file_id from session
    file_id = request.session.get('file', None)
    if not file_id:
        return HttpResponse("No file ID found in session", status=400)
    
    # Retrieve the data file object
    file_model = get_object_or_404(DataFile, file_id=file_id)
    
    # Load the data using your custom method (assuming it's returning a pandas DataFrame)
    data = file_model.load_file()
    
    if data.empty:
        return HttpResponse("No data available", status=400)

    # Calculate various statistics
    data_summary = {
        'columns': list(data.columns),
        'missing_values': data.isnull().sum().to_dict(),  # Missing values per column
        'mean': data.mean(numeric_only=True).to_dict(),  # Mean of numeric columns
        'median': data.median(numeric_only=True).to_dict(),  # Median of numeric columns
        'description': data.describe(include='all').to_dict(),  # Summary statistics for all columns
    }

    # Option 1: Return as JSON (can be used for API response)
    if request.is_ajax():  # Check if it's an AJAX request (to return JSON)
        return JsonResponse(data_summary)

    # Option 2: Return as HTML (for rendering in a template)
    return render(request, 'data_details.html', {'data_summary': data_summary})