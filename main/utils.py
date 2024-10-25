import os
import json
import uuid
import pickle
from django.conf import settings
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.utils import PlotlyJSONEncoder
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

def format_predictions(nums):
    """
    Round the list of predictions to 3 decimal places
    Return only the first 100 predictions
    """
    return [round(num, 3) for num in nums][:100]

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

def plot_kmeans_clusters(X, labels, centroids, features, x_feature, y_feature):
    """Plot KMeans clusters and centroids using Plotly"""
    
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
        color_continuous_scale="twilight",
        labels=dict(x="Features", y="Features", color="Correlation")
    )
    # fig.show()
    return json.dumps(fig, cls=PlotlyJSONEncoder)
