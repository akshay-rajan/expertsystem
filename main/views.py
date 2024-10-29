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

from .utils import construct_line, format_predictions, serialize, regression_evaluation, classification_evaluation
from .utils import plot_feature_importances, plot_decision_tree, plot_dendrogram, plot_kmeans_clusters
from .models import MLModel


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
        dataset = request.session.get('file', None)        
        df = pd.DataFrame.from_dict(dataset)
                        
        # Features and Target selection
        features = request.POST.getlist('features')
        target = request.POST.get('target')
        X, y = df[features], df[target]

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # ! Model Building
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # ! Model Evaluation
        y_pred = model.predict(X_test)
        # MSE, RMSE, MAE, R2
        metrics = regression_evaluation(y_test, y_pred)
        
        # Construct the line equation for the model
        intercept = model.intercept_
        coefficients = model.coef_
        equation = construct_line(intercept, coefficients, X, target)
        
        # # Serialize the model and return the download link, store link in session
        # download_link = serialize(model, 'linear_regression')
        # request.session['model'] = download_link
        ml_model = MLModel()
        ml_model.save_model(model)
        request.session['model'] = str(ml_model.model_id)        
        
        return render(request, 'main/linear_regression.html', {
            'actual': y_test[:100],
            'predicted': format_predictions(y_pred), # Round to 3 decimal places, show only 100
            'features': features,
            'target': target,
            'metrics': metrics,
            'line': equation,
            # 'download': download_link,
            'download': "",
        })
        
    # Render the Input Form
    return render(request, 'main/input.html')
    
def lasso(request):
    """Implement Lasso Regression"""
    
    if request.method == 'POST':
        dataset = request.session.get('file', None)   
        df = pd.DataFrame.from_dict(dataset)
        
        features = request.POST.getlist('features')
        target = request.POST.get('target')
        alpha = float(request.POST.get('alpha'))
                
        X, y = df[features], df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = Lasso(alpha=alpha)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        coeff = pd.Series(model.coef_, index=features)

        intercept = model.intercept_
        coefficients = model.coef_
        equation = construct_line(intercept, coefficients, X, target)
        
        metrics = regression_evaluation(y_test, y_pred)
        
        download_link = serialize(model, 'lasso')
        request.session['model'] = download_link
        
        return render(request, 'main/lasso.html', {
            'coefficients': coeff,
            'actual': y_test[:100],
            'predicted': format_predictions(y_pred),
            'features': features,
            'target': target,
            'metrics': metrics,
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
        dataset = request.session.get('file', None)        
        df = pd.DataFrame.from_dict(dataset)
        
        features = request.POST.getlist('features')
        target = request.POST.get('target')
        alpha = float(request.POST.get('alpha'))
                
        X, y = df[features], df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        alpha = float(request.POST.get('alpha'))
                        
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        coeff = pd.Series(model.coef_, index=features)

        intercept = model.intercept_
        coefficients = model.coef_
        equation = construct_line(intercept, coefficients, X, target)
        
        metrics = regression_evaluation(y_test, y_pred)
        
        download_link = serialize(model, 'ridge')
        request.session['model'] = download_link
        
        return render(request, 'main/ridge.html', {
            'coefficients': coeff,
            'actual': y_test[:100],
            'predicted': format_predictions(y_pred),
            'features': features,
            'target': target,
            'metrics': metrics,
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
        dataset = request.session.get('file', None)        
        df = pd.DataFrame.from_dict(dataset)
        
        features = request.POST.getlist('features')
        target = request.POST.get('target')
        
        X, y = df[features], df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = DecisionTreeRegressor(random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        metrics = regression_evaluation(y_test, y_pred)
        
        download_link = serialize(model, 'decision_tree_regression')
        request.session['model'] = download_link
        
        return render(request, 'main/decision_tree_regression.html', {
            'actual': y_test[:100],
            'predicted': format_predictions(y_pred),
            'features': features,
            'target': target,
            'metrics': metrics,
            'download': download_link,
        })        
    
    return render(request, 'main/input.html')
    
def random_forest_regression(request):
    """Random Forest Regressor"""
    
    if request.method == "POST":
        dataset = request.session.get('file', None)        
        df = pd.DataFrame.from_dict(dataset)
        
        features = request.POST.getlist('features')
        target = request.POST.get('target')
        n_estimators = int(request.POST.get('n_estimators'))
                
        X, y = df[features], df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        metrics = regression_evaluation(y_test, y_pred)
        
        download_link = serialize(model, 'random_forest_regression')
        request.session['model'] = download_link
        
        return render(request, 'main/random_forest_regression.html', {
            'actual': y_test[:100],
            'predicted': format_predictions(y_pred),
            'features': features,
            'target': target,
            'metrics': metrics,
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
        dataset = request.session.get('file', None)        
        df = pd.DataFrame.from_dict(dataset)
        
        features = request.POST.getlist('features')
        target = request.POST.get('target')
        n_neighbors = int(request.POST.get('n_neighbors'))
                
        X, y = df[features], df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        metrics = classification_evaluation(y_test, y_pred)
        
        download_link = serialize(model, 'knn')
        request.session['model'] = download_link
        
        return render(request, 'main/knn.html', {
            'actual': y_test[:100],
            'predicted': y_pred[:100],
            'features': features,
            'target': target,
            'metrics': metrics,
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
        dataset = request.session.get('file', None)        
        df = pd.DataFrame.from_dict(dataset)
        
        features = request.POST.getlist('features')
        target = request.POST.get('target')
                
        X, y = df[features], df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = LogisticRegression()
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        metrics = classification_evaluation(y_test, y_pred)
        
        download_link = serialize(model, 'logistic_regression')
        request.session['model'] = download_link
        
        return render(request, 'main/logistic_regression.html', {
            'actual': y_test[:100],
            'predicted': y_pred[:100],
            'features': features,
            'target': target,
            'metrics': metrics,
            'download': download_link,
        })
    
    return render(request, 'main/input.html')

def naive_bayes(request):
    """Gaussian Naive Bayes Classifier"""
    
    if request.method == "POST":
        dataset = request.session.get('file', None)        
        df = pd.DataFrame.from_dict(dataset)
        
        features = request.POST.getlist('features')
        target = request.POST.get('target')
                
        X, y = df[features], df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = GaussianNB()
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_pred_modified = [round(i, 3) for i in y_pred]
        metrics = classification_evaluation(y_test, y_pred)
        
        download_link = serialize(model, 'naive_bayes')
        request.session['model'] = download_link
        
        mean_per_class = [[round(float(j), 4) for j in i] for i in model.theta_]
        
        return render(request, 'main/naive_bayes.html', {
            'actual': y_test[:100],
            'predicted': y_pred_modified[:100],
            'features': features,
            'target': target,
            'results': dict(zip(model.classes_, mean_per_class)),
            'metrics': metrics,
            'download': download_link,
        })
    
    return render(request, 'main/input.html')

def decision_tree(request):
    """Decision Tree Classifier"""
    
    if request.method == "POST":
        dataset = request.session.get('file', None)        
        df = pd.DataFrame.from_dict(dataset)
        
        features = request.POST.getlist('features')
        target = request.POST.get('target')
                
        X, y = df[features], df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        metrics = classification_evaluation(y_test, y_pred)
        
        download_link = serialize(model, 'decision_tree')
            
        graph_json = plot_decision_tree(model, features)
        
        return render(request, 'main/decision_tree.html', {
            'actual': y_test[:100],
            'predicted': y_pred[:100],
            'features': features,
            'target': target,
            'metrics': metrics,
            'download': download_link,
            'tree': graph_json,
        })
    
    return render(request, 'main/input.html')

def random_forest(request):
    """Random Forest Classifier"""
    
    if request.method == "POST":
        dataset = request.session.get('file', None)        
        df = pd.DataFrame.from_dict(dataset)
        
        features = request.POST.getlist('features')
        target = request.POST.get('target')
        n_estimators = int(request.POST.get('n_estimators'))
                
        X, y = df[features], df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        metrics = classification_evaluation(y_test, y_pred)
        
        download_link = serialize(model, 'random_forest')
        request.session['model'] = download_link

        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        graph_json = plot_feature_importances(features, importances, indices)
        
        return render(request, 'main/random_forest.html', {
            'actual': y_test[:100],
            'predicted': y_pred[:100],
            'features': features,
            'target': target,
            'metrics': metrics,
            'download': download_link,
            'graph': graph_json,
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
        dataset = request.session.get('file', None)        
        df = pd.DataFrame.from_dict(dataset)
        
        features = request.POST.getlist('features')
        target = request.POST.get('target')
        kernel = request.POST.get('kernel')
        C = float(request.POST.get('C'))
        
        X, y = df[features], df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = SVC(kernel=kernel, C=C, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        metrics = classification_evaluation(y_test, y_pred)
        
        download_link = serialize(model, 'svm')
        request.session['model'] = download_link
        
        return render(request, 'main/svm.html', {
            'actual': y_test[:100],
            'predicted': y_pred[:100],
            'features': features,
            'target': target,
            'metrics': metrics,
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
        dataset = request.session.get('file', None)
        df = pd.DataFrame.from_dict(dataset)
        
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
        
        download_link = serialize(model, 'kmeans')
        request.session['model'] = download_link

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
            'download': download_link,
            'plot': plot_json,
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
        dataset = request.session.get('file', None)
        df = pd.DataFrame.from_dict(dataset)
        
        features = request.POST.getlist('features')                
        n_clusters = int(request.POST.get('n_clusters'))
        
        X = df[features]

        model = AgglomerativeClustering(n_clusters=n_clusters)
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
        
        download_link = serialize(model, 'hierarchical_clustering')
        request.session['model'] = download_link
        
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
    response['Content-Disposition'] = f'attachment; filename="model-{model_id}.pkl"'
    
    return response

# ? API Endpoints
@csrf_exempt
def predict(request):
    """Open an endpoint to predict using a saved model"""
    if request.method == "POST":
        try:            
            # Load the model
            model_path = request.session.get('model')[1:] # Remove the leading '/'
            if not model_path:
                return JsonResponse({'error': 'No model available'}, status=400)
            
            with open(model_path, 'rb') as file:
                model = pickle.load(file)
            
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
    """Save the uploaded file content to the session as a dict using pandas"""
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

        
        # Store the file name and dict in the session
        request.session['filename'] = file.name
        request.session['file'] = df.to_dict()

        return JsonResponse({'message': 'File uploaded successfully!'})
    
    return JsonResponse({'error': 'Invalid request method'}, status=400)

def get_file(request):
    """Return the file content stored in the session"""
    if request.method == 'POST':
        file_dict = request.session.get('file', None)
        filename = request.session.get('filename', 'file.csv')
    
        if file_dict:
            df = pd.DataFrame.from_dict(file_dict)
            columns = df.columns.tolist()
            
            correlation_matrix = df.corr()

            return JsonResponse({
                'filename': filename, 
                'file': file_dict,
                'columns': columns,
                'correlation_matrix': correlation_matrix.to_dict(),
            })
        return JsonResponse({'Error': 'No file available'}, status=400)
    return JsonResponse({'error': 'Invalid request method'}, status=400)

# ? Preprocessing

def preprocessing(request):
    context = {}

    if request.method == 'POST' and request.FILES.get('file'):
        uploaded_file = request.FILES['file']
        try:
            # Read the uploaded file into a DataFrame
            data = pd.read_csv(uploaded_file)

            
            data.replace(0, np.nan, inplace=True)
            # Store the initial dataset in the session
            request.session['updated_data'] = data.to_dict()

            null_columns = data.columns[data.isnull().any()]
            non_numerical_cols = data.select_dtypes(include=['object', 'category']).columns

            ################################################################################

            # print("TESTin here")
            # print(data.info())
            print(type(data.describe()))
            print(data.describe())

            # print(type(data.head(10)))

           

            # print(data.isnull().sum())


            #################################################################################

            # Prepare the data preview for rendering
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
        except Exception as e:
            return JsonResponse({'error': f"Error processing data: {e}"})

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
        
        if not missing_value_strategy or not selected_columns:
            return JsonResponse({'error': 'Invalid input, strategy and columns are required'}, status=400)

        
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

        if not encoding_strategy or not encoding_columns:
            return JsonResponse({'error': 'Invalid input, strategy and columns are required'}, status=400)


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
  
def scaling(request):
    if request.method == 'POST':
        # Load the updated data from session
        data_dict = request.session.get('updated_data')
        if not data_dict:
            return JsonResponse({'error': 'No data available'}, status=400)

        # Convert data dictionary to pandas DataFrame
        data = pd.DataFrame.from_dict(data_dict)

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
        request.session['updated_data'] = data.to_dict()

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

