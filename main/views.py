import csv
import json
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
from django.http import JsonResponse

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
            # print(data.describe())
           

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

        data = pd.DataFrame.from_dict(data_dict)
        databody = json.loads(request.body)

        scaling_strategy = databody.get('strategy')
        scaling_columns = databody.get('columns')
        print(scaling_columns)

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

