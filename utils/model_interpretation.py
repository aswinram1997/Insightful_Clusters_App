import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import shap
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def perform_pca(dataset, labels):
    """
    Performs Principal Component Analysis (PCA) on the dataset.

    Parameters:
    - dataset (pd.DataFrame): The dataset to perform PCA on.
    - labels (np.ndarray): The cluster labels.

    Returns:
    - pca_result (np.ndarray): The result of PCA transformation.
    """
    combined_data = pd.concat([dataset, pd.DataFrame(labels, columns=['labels'])], axis=1)
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(combined_data)
    return pca_result

def create_pca_chart(pca_result, labels):
    """
    Creates a 3D scatter plot of the PCA results with color-coded clusters.

    Parameters:
    - pca_result (np.ndarray): The result of PCA transformation.
    - labels (np.ndarray): The cluster labels.

    Returns:
    - None
    """
    pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2', 'PC3'])
    pca_df['Cluster'] = labels.astype(str)
    clusters = pca_df['Cluster'].unique()

    fig = go.Figure()
    num_clusters = len(clusters)
    colors = px.colors.qualitative.Prism[:num_clusters]
    color_palette = colors[:num_clusters]

    for cluster, color in zip(clusters, color_palette):
        fig.add_trace(go.Scatter3d(
            x=pca_df.loc[pca_df['Cluster'] == cluster, 'PC1'],
            y=pca_df.loc[pca_df['Cluster'] == cluster, 'PC2'],
            z=pca_df.loc[pca_df['Cluster'] == cluster, 'PC3'],
            mode='markers',
            marker=dict(color=color),
            name=cluster
        ))

    fig.update_layout(title='3D PCA Chart', scene=dict(xaxis_title='PC1', yaxis_title='PC2', zaxis_title='PC3'))
    fig.show()

def apply_shap(dataset, labels, model):
    """
    Applies SHAP (SHapley Additive exPlanations) on the clustered dataset.

    Parameters:
    - dataset (pd.DataFrame): The dataset to apply SHAP on.
    - labels (np.ndarray): The cluster labels.
    - model: The model used for SHAP analysis.

    Returns:
    - shap_values (np.ndarray): The SHAP values.
    """
    # Create a DataFrame with cluster labels and original dataset features
    dataset_with_labels = dataset.copy()
    dataset_with_labels['labels'] = labels
    
    # Create the SHAP explainer using the modified dataset
    explainer = shap.Explainer(model, data=dataset_with_labels)
    
    # Compute the SHAP values
    shap_values = explainer.shap_values(dataset_with_labels)
    
    return shap_values, dataset_with_labels

