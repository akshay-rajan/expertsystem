import csv
import os
import uuid
import pickle
import numpy as np
import pandas as pd
from django.conf import settings
from django.shortcuts import redirect, render
from django.core.files.storage import FileSystemStorage
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from django.http import HttpResponse

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
            df = pd.read_csv(dataset)
        else:
            df = pd.read_excel(dataset)
        
        features = request.POST.getlist('features')
        target = request.POST.get('target')
                
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
        
        return render(request, 'main/results.html', {
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
    train_set = ["California Housing (80%)"]
    test_set = ["California Housing (20%)"]
    
    return render(request, 'main/input.html', {
        'train_set': train_set, 
        'test_set': test_set,
    })

def samples(request):
    return render(request, 'main/samples.html')

# ? Helper Functions

# Preprocessing part ->Deepu
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

def preprocessing(request):
    context = {}


    if request.method == 'POST' and request.FILES.get('file'):
        
        uploaded_file = request.FILES['file']
        try:
            # Read the uploaded file into a DataFrame
            data = pd.read_csv(uploaded_file)

          

        # Get the preprocessing options from the form
            missing_value_strategy = request.POST.get('missing_value_strategy')
            selected_columns = request.POST.getlist('feature_selection')
            encoding_strategy = request.POST.get('encoding_strategy')
            encoding_columns = request.POST.getlist('encoding_selection')
            scaling_strategy = request.POST.get('scaling_strategy')

            
            # Handle missing values based on the strategy and selected columns
            if missing_value_strategy and selected_columns:
                if missing_value_strategy == 'mean':
                    for col in selected_columns:
                        if data[col].dtype != 'object':  # Ensure column is numerical
                            data[col].fillna(data[col].mean(), inplace=True)
                elif missing_value_strategy == 'median':
                    for col in selected_columns:
                        if data[col].dtype != 'object':  # Ensure column is numerical
                            data[col].fillna(data[col].median(), inplace=True)
                elif missing_value_strategy == 'drop':
                    data.dropna(subset=selected_columns, inplace=True)

            # Handle encoding
            if encoding_strategy == 'onehot' and encoding_columns:
                data = pd.get_dummies(data, columns=encoding_columns)
            elif encoding_strategy == 'label' and encoding_columns:
                le = LabelEncoder()
                for col in encoding_columns:
                    if data[col].dtype == 'object':  # Ensure column is categorical
                        data[col] = le.fit_transform(data[col])

            # Handle scaling
            if scaling_strategy == 'standard':
                scaler = StandardScaler()
                data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
            elif scaling_strategy == 'normalize':
                scaler = MinMaxScaler()
                data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

            # Store the updated dataset back in session
            request.session['updated_data'] = data.to_dict()

            

            # Show the updated data preview
            context['data_preview'] = data.to_html(classes='table table-bordered table-hover table-striped', index=False)

        except Exception as e:
            context['error'] = f"Error processing data: {e}"

    # No data in session, show upload form
    return render(request, 'main/preprocessing.html', context)

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

