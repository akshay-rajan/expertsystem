import os
import pickle
import uuid
from django.conf import settings
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def construct_line(intercept, coefficients, X, target):
    """Given the coefficients and intercept, construct the line equation as a string"""
    equation = f"{target} = {intercept:.2f}"
    for feature, coef in zip(X.columns, coefficients):
        if round(coef, 2) == 0: 
            continue
        if coef > 0:
            equation += f" + ({coef:.2f} * {feature})"
        else:
            equation += f" - ({abs(coef):.2f} * {feature})"
    return equation

def serialize(model, algorithm):
    """Serialize the model and save it to a .pkl file, return the path"""
    model_filename = f"{algorithm}_{uuid.uuid4().hex[:6]}.pkl"
    model_path = os.path.join(settings.MEDIA_ROOT, model_filename)
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)
    download_link = os.path.join(settings.MEDIA_URL, model_filename)
    return download_link

def regression_evaluation(y_test, y_pred):
    """Perform evaluations of a regression model"""
    mse = mean_squared_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return {
        'mse': round(mse, 2),
        'rmse': round(rmse, 2),
        'mae': round(mae, 2),
        'r2': round(r2, 2)
    }

def classification_evaluation(y_test, y_pred):
    """Perform evaluations of a classification model"""
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    return {
        'accuracy': round(accuracy, 2),
        'precision': round(precision, 2),
        'recall': round(recall, 2),
        'f1': round(f1, 2)
    }



