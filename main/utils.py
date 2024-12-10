import os
import json
from django.conf import settings
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.utils import PlotlyJSONEncoder
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def get_input(request_post, *args):
    """Get the parameters entered by the user"""
    
    inputs = (request_post.getlist('features'), request_post.get('target'), float(request_post.get('testsize')))
    
    for arg in args:
        if isinstance(arg, tuple):
            param_name, default_value = arg
            value = request_post.get(param_name, default_value)
        else:
            param_name = arg
            value = request_post.get(param_name)
        inputs += (value,)
    
    return inputs

def list_available_datasets(group=None):
    """List the available datasets to choose from"""
    datasets = [
        {
            "name": "California Housing",
            "file": "fetch_california_housing.xlsx",
            "type": "XLSX",
            "note": "For Regression (Large)",
            "group": {"regression"}
        },
        {
            "name": "California Housing",
            "file": "fetch_california_housing.csv",
            "type": "CSV",
            "note": "For Regression (Large)",
            "group": {"regression"}
        },
        {
            "name": "Numerical Data",
            "file": "numerical_data.xlsx",
            "type": "XLSX",
            "note": "For Regression (Large)",
            "group": {"regression"}
        },
        {
            "name": "Iris",
            "file": "iris.csv",
            "type": "CSV",
            "note": "For Classification (Small)",
            "group": {"classification", "clustering"}
        },
        {
            "name": "Mall Customers",
            "file": "mall_customers.csv",
            "type": "CSV",
            "note": "For Clustering (Small)",
            "group": {"clustering"}
        },
        {   
            "name": "Countries and Purchases",
            "file": "purchases.csv",
            "type": "CSV",
            "note": "Uncleaned (Small)",
            "group": {"preprocessing"}
        },
        {
            "name": "Pima Indians Diabetes",
            "file": "diabetes.csv",
            "type": "CSV",
            "note": "For Classification (Binary)",
            "group": {"classification"}
        },
        {
            "name": "Big Mart Sales",
            "file": "big_mart_sales.csv",
            "type": "CSV",
            "note": "Uncleaned (Large)",
            "group": {"preprocessing"}
        },
        {
            "name": "Hospital Charges",
            "file": "insurance.csv",
            "type": "CSV",
            "note": "Uncleaned",
            "group": {"preprocessing"}
        },
        {
            "name": "Hospital Charges",
            "file": "insurance_processed.csv",
            "type": "CSV",
            "note": "For Regression",
            "group": {"regression"}
        }
    ]
    # If no group is specified, return all datasets, for 'sample datasets'
    if group is None:
        return datasets
    # Otherwise, filter the datasets based on the group specified, for 'input'
    files = [dataset['file'] for dataset in datasets if group in dataset['group']]
    return files

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

def format_predictions(nums):
    """
    Round the list of predictions to 3 decimal places
    Return only the first 100 predictions
    """
    return [round(num, 3) for num in nums][:100]

def regression_evaluation(y_test, y_pred):
    """Perform evaluations of a regression model"""
    mse = mean_squared_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return {
        'mse': round(mse, 4),
        'rmse': round(rmse, 4),
        'mae': round(mae, 4),
        'r2': round(r2, 4)
    }

def classification_evaluation(y_test, y_pred):
    """Perform evaluations of a classification model"""
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    return {
        'accuracy': round(accuracy, 4) * 100,
        'precision': round(precision, 4) * 100,
        'recall': round(recall, 4) * 100,
        'f1': round(f1, 4) * 100
    }

def generate_preview_response(data):
    """Helper function to prepare data preview."""
    null_columns = data.columns[data.isnull().any()]
    non_numerical_cols = data.select_dtypes(include=['object', 'category']).columns

    return {
        'json_data': data.head(20).to_json(orient='records'),
        'headers': data.columns.tolist(),
        'null_columns': null_columns.tolist(),
        'non_numerical_cols': non_numerical_cols.tolist()
    }

def plot_feature_importances(features, importances, indices):
    """Plot the feature importances for Random Forest""", 
    fig = px.bar(
        x=[features[int(i)] for i in indices],
        y=importances[indices],
        labels={'x': "Features", 'y': "Importance"},
        title='Feature Importances',
        template='plotly_white'
    )
    fig.update_traces(marker_color='rgb(0,150,255)')
    return json.dumps(fig, cls=PlotlyJSONEncoder)

def plot_decision_tree(model, feature_names):
    """Plot a decision tree using Plotly Treemap"""
    
    # Initialize labels and parents with size of nodes in the tree
    labels = [''] * model.tree_.node_count
    parents = [''] * model.tree_.node_count
    
    # Root node is labeled as 'root'
    labels[0] = 'root'
    
    # Iterate through the tree nodes
    for i, (f, t, l, r) in enumerate(zip(
        model.tree_.feature,
        model.tree_.threshold,
        model.tree_.children_left,
        model.tree_.children_right,
    )):
        if l != r:  # If the node has children (non-leaf node)
            # Label left child with the condition for the split
            labels[l] = f'{feature_names[f]} <= {t:g}'
            # Label right child with the condition for the split
            labels[r] = f'{feature_names[f]} > {t:g}'
            # Set both left and right children's parent to the current node
            parents[l] = parents[r] = labels[i]
    
    # Create the Treemap plot using Plotly
    fig = go.Figure(go.Treemap(
        branchvalues='total',
        labels=labels,
        parents=parents,
        values=model.tree_.n_node_samples,  # Node sizes based on number of samples
        textinfo='label+value+percent root',  # Display label, value, and % relative to the root
         # Colors based on node impurity
        marker=dict(
            colors=model.tree_.impurity,
            colorscale='thermal',
            cmin=model.tree_.impurity.min(),
            cmax=model.tree_.impurity.max(),
        ),
        customdata=list(map(str, model.tree_.value)),  # Class distribution in custom data
        hovertemplate='''<b>%{label}</b><br>
        impurity: %{color}<br>
        samples: %{value} (%{percentRoot:%.2f})<br>
        value: %{customdata}''',
    ))
    
    # Return the Plotly figure in JSON format to be used in the frontend
    return json.dumps(fig, cls=PlotlyJSONEncoder)

def plot_dendrogram(linkage_matrix, labels):
    """Plot a dendrogram using Plotly"""
    if len(labels) != linkage_matrix.shape[0] + 1:
        raise ValueError("Number of labels must match the number of observations in the data.")
    fig = ff.create_dendrogram(
        linkage_matrix,
        # labels=labels,
    )
    fig.update_layout(title='Dendrogram', template='plotly_white', width=1000, height=600)
    return json.dumps(fig, cls=PlotlyJSONEncoder)

def plot_clusters(X, labels, centroids, features, x_feature, y_feature):
    """Plot clusters and centroids using Plotly"""
    
    # Create a scatter plot for the data points
    fig = go.Figure()
    
    # Plot the clusters
    fig.add_trace(go.Scatter(
        x=X[:, x_feature],
        y=X[:, y_feature],
        mode='markers',
        marker=dict(
            size=10,
            color=labels,  # Color based on cluster labels
            colorscale='Viridis',
            line=dict(width=1, color='DarkSlateGrey'),
        ),
        name="Data Points"
    ))

    # Plot the centroids
    fig.add_trace(go.Scatter(
        x=centroids[:, x_feature],
        y=centroids[:, y_feature],
        mode='markers',
        marker=dict(
            size=12,
            color='red',
            symbol='x',
            line=dict(width=2, color='red'),
        ),
        name="Centroids"
    ))
    
    # Update layout
    fig.update_layout(
        title="Clusters and Centroids",
        xaxis_title=features[x_feature],
        yaxis_title=features[y_feature],
        template="plotly_white",
        showlegend=True
    )
    
    # Return the JSON of the plot
    return json.dumps(fig, cls=PlotlyJSONEncoder)

def plot_heatmap(correlation_matrix):
    """Plot Correlation Heatmap from the session"""
    fig = px.imshow(
        correlation_matrix, 
        color_continuous_scale="RdBu",
        range_color=[-1, 1],
    )
    fig.update_layout(
        template="plotly_white",
        width=450,
        height=450,
        coloraxis_showscale=False
    )
    return json.dumps(fig, cls=PlotlyJSONEncoder)

def plot_scatter(df, x_label, y_label):
    """Plot Scatter plot from the session"""
    fig = px.scatter(
        df, 
        x=x_label, 
        y=y_label, 
        template='plotly_white',
        title=f"{y_label} vs {x_label}"
    )
    # Use different colors for markers
    fig.update_traces(marker={
        'size': 10,
        'line': {'width': 1, 'color': 'DarkSlateGrey'},
        'color': 'rgb(0,150,255)',
        'opacity': 0.6
    })    
    
    return json.dumps(fig, cls=PlotlyJSONEncoder)
