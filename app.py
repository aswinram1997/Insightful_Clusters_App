import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import shap
import streamlit as st

from utils.datasets import load_dataset
from utils.data_preprocessing import extract_column_types
from utils.data_preprocessing import handle_missing_values
from utils.data_preprocessing import handle_date_feature
from utils.data_preprocessing import encode_categorical_features
from utils.data_preprocessing import normalize_dataset
from utils.clustering import define_k_values_range
from utils.clustering import kmeans_clustering, agglomerative_clustering
from utils.clustering import evaluate_clustering, perform_clustering_evaluation
from utils.clustering import count_votes, select_best_combination
from utils.clustering import perform_final_clustering
from utils.model_training import train_xgboost_model, calculate_f1_score
from utils.model_interpretation import perform_pca, create_pca_chart, apply_shap


# Set Title
st.title("Insightful Clusters✨")
st.caption("Insightful Clusters revolutionizes cluster segmentation and interpretation for any dataset. Our seamless and user-friendly solution leverages cutting-edge ML algorithms to automatically identify distinct groups or clusters in your data. Simply upload your dataset, and let the app handle the entire clustering process. Visualize and interpret the clusters effortlessly, gaining valuable insights. Whether you're analyzing customer data, market research, or any dataset, Insightful Clusters streamlines cluster segmentation and interpretation, empowering you to make data-driven decisions effortlessly.")

# Add a sidebar for navigation
st.sidebar.title("Navigation")
algorithm = st.sidebar.selectbox("Please choose your strategy:", ["Automated Clustering", "Manual Clustering"])

# dataset options
dataset_options = ['Diabetes Patient Clustering', 'Automobile Customer Segmentation', 'Mall Shopper Segmentation', 'Blueberry Growth Characteristics', 'Sleep, Health & Lifestyle', 'Iris Flowers Clustering', 'Wine Varieties Clustering', 'Custom']



# Automated Clustering
if algorithm == "Automated Clustering":
    try:
        # Step 1: Load the preprocessed dataset
        st.subheader("*Load Dataset*")
        st.caption("Choose a dataset for clustering analysis from predefined options like Iris, Wine, Mall Customer, or upload your own custom dataset in Excel format. Explore and analyze the dataset to uncover valuable insights and patterns relevant to your specific data. This step sets the foundation for further analysis and clustering.")
        st.markdown("<br>", unsafe_allow_html=True)

        # List of available dataset options
        dataset_options = dataset_options

        # Display dataset selection dropdown
        selected_dataset = st.selectbox('Select a dataset', dataset_options)

        dataset = None  # Initialize dataset variable

        if selected_dataset == 'Custom':
            dataset = load_dataset(selected_dataset)
            # Use the custom dataset in your code
            if dataset is not None:
                # Replace this with your code to process the custom dataset
                st.caption('Custom dataset uploaded')
        else:
            # Load the preprocessed dataset
            dataset = load_dataset(selected_dataset)

        if dataset is not None:  # Check if dataset is loaded before proceeding to the next steps
            try:
                # Step 2: Extract column types from dataset
                numerical_columns, categorical_columns, date_columns = extract_column_types(dataset)
                st.write("Numerical Features Identified:", numerical_columns)
                st.write("Categorical Features Identified:", categorical_columns)
                st.write("Date Features Identified:", date_columns)

                # Step 3: Perform data preprocessing
                dataset = handle_missing_values(dataset, numerical_columns, categorical_columns, date_columns)
                dataset = handle_date_feature(dataset, date_columns)
                dataset = normalize_dataset(dataset, numerical_columns)
                dataset, encoded_columns = encode_categorical_features(dataset, categorical_columns)

                # Step 4: Define the range of k values for clustering
                k_values_range = define_k_values_range()

                # Step 5: Perform clustering and evaluation for different algorithms and k values
                evaluation_scores = perform_clustering_evaluation(dataset, k_values_range)

                # Step 6: Count votes for each algorithm and k value combination
                votes = count_votes(evaluation_scores)

                # Step 7: Select the best combination based on majority voting
                best_combination = select_best_combination(votes, evaluation_scores)
   
                # Step 8: Perform final clustering with the best algorithm and k value combination
                algorithm, k = best_combination
                labels = perform_final_clustering(dataset, algorithm, k)
   
                # Step 9: Evaluate the clustering results using Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Index
                silhouette, db, ch = evaluate_clustering(dataset, labels)
      
                # Step 10: Split the dataset into training and testing sets (80/20 split)
                X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.2, random_state=42)

                # Step 11: Train an XGBoost model on the training set
   
                # Step 12: Calculate the F1 score on the testing set
                f1 = calculate_f1_score(model, X_test, y_test)
   
                # Step 13: Perform Principal Component Analysis (PCA) on the dataset for visualization
                pca_result = perform_pca(dataset, labels)

                # Step 14: Apply SHAP on the clustered dataset to interpret the meaning of each cluster
                shap_values = apply_shap(dataset, labels, model)


            except Exception as e:
                 st.error("For successful dataset loading, consider the following steps:"
         "\n\n1. Clean your data: Identify and resolve inconsistencies, errors, and missing values."
         "\n\n2. Verify data types: Ensure correct assignment of numeric, categorical, and datetime data types."
         "\n\n3. Remove long string variables: Consider removing features with excessively long strings."
         "\n\nTaking these steps will increase the likelihood of smoothly loading the dataset. " + str(e))
        else:
            st.caption('Please select a dataset to proceed.')

    except Exception as e:
        st.error("Error in loading dataset"+ str(e))
       


    st.write("---")  # Add a horizontal line for visual separation

    # Step 15: Display the results
    
    # Algorithm Performance Dashboard
    try:
        st.subheader("*Algorithm Performance Dashboard*")
        st.caption("Explore our carefully chosen clustering algorithm and K value selection, which have demonstrated superior results on your dataset by utilizing evaluation metrics such as Silhouette Score, DB Index, and CH Index. These metrics comprehensively evaluate the effectiveness of the clustering algorithm in terms of cluster separation, compactness, and cohesion.")
       
        
        if dataset is not None:
            # Set up the layout
            col1 = st.columns(4)

            with col1[0]:
                st.markdown('Clustering Algorithm')
                st.info(f'{algorithm} (K={k})')

            with col1[1]:
                st.markdown('Silhouette Score')
                st.info(silhouette)

            with col1[2]:
                st.write('DB Index')
                st.info(db)

            with col1[3]:
                st.markdown('CH Index')
                st.info(ch)


            # Set up the layout
            col1 = st.columns(2)

            with col1[0]:
                st.write('Supervised Algorithm')
                st.info("XgBoost")

            with col1[1]:
                st.write('F1 Score')
                st.info(round(f1,2))

    except Exception as e:
        st.error("Error generating Algorithm Performance Dashboard")



    st.write("---")  # Add a horizontal line for visual separation



    try:
        # Create PCA Chart
        st.subheader("*Cluster Visualization with PCA*")
        st.caption("Gain insights into cluster relationships and separations by examining the dimensionally reduced 3D scatter plot. Each cluster is represented by a different color, providing information about their distribution and proximity, allowing you to visualize the spatial arrangement and understand the distinct groupings within your data.")

        if dataset is not None:
            pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2', 'PC3'])
            pca_df['Cluster'] = labels.astype(str)

            # Get the unique cluster labels
            clusters = pca_df['Cluster'].unique()
            # Sort the cluster labels in ascending order
            clusters = sorted(clusters)

            # Create the 3D scatter plot with color-coded clusters
            fig = go.Figure()

            # Generate a color palette for the given number of clusters using Plotly
            num_clusters = len(clusters)
            colors = px.colors.qualitative.Prism[:num_clusters]
            color_palette = colors[:num_clusters]

            for cluster, color in zip(clusters, color_palette):
                cluster_data = pca_df.loc[pca_df['Cluster'] == cluster]
                fig.add_trace(
                    go.Scatter3d(
                        x=cluster_data['PC1'],
                        y=cluster_data['PC2'],
                        z=cluster_data['PC3'],
                        mode='markers',
                        marker=dict(color=color, size=12, line=dict(width=1)),  # Adjust the marker size and line width
                        name=cluster
                    )
                )

            # Calculate the range for each axis
            x_range = [pca_df['PC1'].min(), pca_df['PC1'].max()]
            y_range = [pca_df['PC2'].min(), pca_df['PC2'].max()]
            z_range = [pca_df['PC3'].min(), pca_df['PC3'].max()]

            # Set the title and axes labels, and adjust the aspect ratio
            fig.update_layout(
                scene=dict(
                    xaxis=dict(range=x_range, autorange=False),
                    yaxis=dict(range=y_range, autorange=False),
                    zaxis=dict(range=z_range, autorange=False),
                    xaxis_title='PC1',
                    yaxis_title='PC2',
                    zaxis_title='PC3',
                    aspectmode='manual',  # Use manual aspect ratio
                    aspectratio=dict(x=1, y=1, z=0.8)  # Adjust the aspect ratio based on your data
                ),
                margin=dict(l=0, r=0, b=0, t=0),
                autosize=True  # Fit the plot to the available space
            )

            # Adjust the legend properties
            fig.update_layout(
                legend=dict(
                    font=dict(size=14),  # Increase the font size
                    yanchor="top",  # Keep the legend in the top-right corner
                    y=0.95,  # Adjust the vertical position
                    xanchor="right",
                    x=0.95  # Adjust the horizontal position
                ),
                legend_traceorder="normal"  # Arrange the legend labels in ascending order
            )

            # Display the plot using Streamlit
            st.plotly_chart(fig)

    except Exception as e:
        st.error("Error in Cluster Visualization with PCA")



    st.write("---")  # Add a horizontal line for visual separation



    try:
        st.subheader("*Cluster Interpretation with SHAP*")
        st.caption("A SHAP summary plot shows the importance of different features for each cluster. The plot ranks the features based on their contribution to each cluster. Features with higher SHAP values on the plot have a stronger influence on that cluster. By looking at the plot, you can understand which features are more important in distinguishing and defining the characteristics of each cluster.")
        
        if dataset is not None:

            # Create a SHAP summary plot

            fig, ax = plt.subplots(figsize=(8, 5))  # Adjust the figsize as needed
            shap.summary_plot(shap_values, dataset)

            # Set the figure background color to transparent
            fig.patch.set_alpha(0.0)

            # Set the axes background color to transparent
            ax.patch.set_alpha(0.0)

            # Remove the border around the plot
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.spines["left"].set_visible(False)

            # Set the ticks and labels color to black
            ax.tick_params(axis='x', colors='black')
            ax.tick_params(axis='y', colors='black')

            # Set the axis labels color to black
            plt.xlabel('Shap Values', color='black')  # Set x-axis title

            # Make the x and y-axis lines bold
            plt.axhline(0, color='black', linewidth=0)  # Horizontal line
            plt.axvline(0, color='black', linewidth=0)  # Vertical line

            # Adjust legend properties and location
            legend = plt.legend(loc='lower right', frameon=False)  # Move the legend to the bottom right corner and remove box outline
            plt.setp(legend.get_texts(), color='black')  # Set the legend text color

            # Adjust the plot layout to fit well in the figure
            plt.tight_layout()

            # Save the modified plot as a transparent image
            fig.savefig('shap_summary_plot.png', transparent=True, bbox_inches='tight')

            # Display the plot using Streamlit
            st.image('shap_summary_plot.png')

    except Exception as e:
        st.error("Error in cluster Interpretation with SHAP")
        
        
        
else:
    algorithm = st.sidebar.selectbox("Select Clustering Algorithm", ["K-Means", "Agglomerative"])
    k = st.sidebar.slider("Select K Value", min_value=3, max_value=7, value=3, step=1)

    try:
        # Step 1: Load the preprocessed dataset
        st.subheader("*Load Dataset*")
        st.caption("Choose a dataset for clustering analysis from predefined options like Iris, Wine, Mall Customer, or upload your own custom dataset in Excel format. Explore and analyze the dataset to uncover valuable insights and patterns relevant to your specific data. This step sets the foundation for further analysis and clustering.")
        st.markdown("<br>", unsafe_allow_html=True)

        # List of available dataset options
        dataset_options = dataset_options

        # Display dataset selection dropdown
        selected_dataset = st.selectbox('Select a dataset', dataset_options)

        dataset = None  # Initialize dataset variable

        if selected_dataset == 'Custom':
            dataset = load_dataset(selected_dataset)
            # Use the custom dataset in your code
            if dataset is not None:
                # Replace this with your code to process the custom dataset
                st.caption('Custom dataset loaded')
        else:
            # Load the preprocessed dataset
            dataset = load_dataset(selected_dataset)

        if dataset is not None:  # Check if dataset is loaded before proceeding to the next steps
            try:
                # Step 2: Extract column types from dataset
                numerical_columns, categorical_columns, date_columns = extract_column_types(dataset)
                st.write("Numerical Features Identified:", numerical_columns)
                st.write("Categorical Features Identified:", categorical_columns)
                st.write("Date Features Identified:", date_columns)

                # Step 3: Perform data preprocessing
                dataset = handle_missing_values(dataset, numerical_columns, categorical_columns, date_columns)
                dataset = handle_date_feature(dataset, date_columns)
                dataset = normalize_dataset(dataset, numerical_columns)
                dataset, encoded_columns = encode_categorical_features(dataset, categorical_columns)

                # Step 8: Perform final clustering with the selected algorithm and k value
                if algorithm == "K-Means":
                    labels = kmeans_clustering(dataset, k)
                elif algorithm == "Agglomerative":
                    labels = agglomerative_clustering(dataset, k)

                # Step 9: Evaluate the clustering results using Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Index
                silhouette, db, ch = evaluate_clustering(dataset, labels)

                # Step 10: Split the dataset into training and testing sets (80/20 split)
                X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.2, random_state=42)

                # Step 11: Train an XGBoost model on the training set
                model = train_xgboost_model(X_train, y_train)

                # Step 12: Calculate the F1 score on the testing set
                f1 = calculate_f1_score(model, X_test, y_test)

                # Step 13: Perform Principal Component Analysis (PCA) on the dataset for visualization
                pca_result = perform_pca(dataset, labels)

                # Step 14: Apply SHAP on the clustered dataset to interpret the meaning of each cluster
                shap_values = apply_shap(dataset, labels, model)

            except Exception as e:
                st.error("Error in algorithm execution")
        else:
            st.caption('Please select a dataset to proceed.')

    except Exception as e:
        st.error("Error in loading dataset")


    st.write("---")  # Add a horizontal line for visual separation

    # Step 15: Display the results

    try:
        st.subheader("*Algorithm Performance Dashboard*")
        st.caption("Explore our carefully chosen clustering algorithm and K value selection, which have demonstrated superior results on your dataset by utilizing evaluation metrics such as Silhouette Score, DB Index, and CH Index. These metrics comprehensively evaluate the effectiveness of the clustering algorithm in terms of cluster separation, compactness, and cohesion.")
        
        if dataset is not None:
            # Set up the layout
            col1 = st.columns(4)

            with col1[0]:
                st.markdown('Clustering Algorithm')
                st.info(f'{algorithm} (K={k})')

            with col1[1]:
                st.markdown('Silhouette Score')
                st.info(silhouette)

            with col1[2]:
                st.write('DB Index')
                st.info(db)

            with col1[3]:
                st.markdown('CH Index')
                st.info(ch)


            # Set up the layout
            col1 = st.columns(2)

            with col1[0]:
                st.write('Supervised Algorithm')
                st.info("XgBoost")

            with col1[1]:
                st.write('F1 Score')
                st.info(round(f1,2))

    except Exception as e:
        st.error("Error generating Algorithm Performance Dashboard")



    st.write("---")  # Add a horizontal line for visual separation



    try:
        # Create PCA Chart
        st.subheader("*Cluster Visualization with PCA*")
        st.caption("Gain insights into cluster relationships and separations by examining the dimensionally reduced 3D scatter plot. Each cluster is represented by a different color, providing information about their distribution and proximity, allowing you to visualize the spatial arrangement and understand the distinct groupings within your data.")
        
        if dataset is not None:
            pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2', 'PC3'])
            pca_df['Cluster'] = labels.astype(str)

            # Get the unique cluster labels
            clusters = pca_df['Cluster'].unique()
            # Sort the cluster labels in ascending order
            clusters = sorted(clusters)

            # Create the 3D scatter plot with color-coded clusters
            fig = go.Figure()

            # Generate a color palette for the given number of clusters using Plotly
            num_clusters = len(clusters)
            colors = px.colors.qualitative.Prism[:num_clusters]
            color_palette = colors[:num_clusters]

            for cluster, color in zip(clusters, color_palette):
                cluster_data = pca_df.loc[pca_df['Cluster'] == cluster]
                fig.add_trace(
                    go.Scatter3d(
                        x=cluster_data['PC1'],
                        y=cluster_data['PC2'],
                        z=cluster_data['PC3'],
                        mode='markers',
                        marker=dict(color=color, size=12, line=dict(width=1)),  # Adjust the marker size and line width
                        name=cluster
                    )
                )

            # Calculate the range for each axis
            x_range = [pca_df['PC1'].min(), pca_df['PC1'].max()]
            y_range = [pca_df['PC2'].min(), pca_df['PC2'].max()]
            z_range = [pca_df['PC3'].min(), pca_df['PC3'].max()]

            # Set the title and axes labels, and adjust the aspect ratio
            fig.update_layout(
                scene=dict(
                    xaxis=dict(range=x_range, autorange=False),
                    yaxis=dict(range=y_range, autorange=False),
                    zaxis=dict(range=z_range, autorange=False),
                    xaxis_title='PC1',
                    yaxis_title='PC2',
                    zaxis_title='PC3',
                    aspectmode='manual',  # Use manual aspect ratio
                    aspectratio=dict(x=1, y=1, z=0.8)  # Adjust the aspect ratio based on your data
                ),
                margin=dict(l=0, r=0, b=0, t=0),
                autosize=True  # Fit the plot to the available space
            )

            # Adjust the legend properties
            fig.update_layout(
                legend=dict(
                    font=dict(size=14),  # Increase the font size
                    yanchor="top",  # Keep the legend in the top-right corner
                    y=0.95,  # Adjust the vertical position
                    xanchor="right",
                    x=0.95  # Adjust the horizontal position
                ),
                legend_traceorder="normal"  # Arrange the legend labels in ascending order
            )

            # Display the plot using Streamlit
            st.plotly_chart(fig)

    except Exception as e:
        st.error("Error in Cluster Visualization with PCA")



    st.write("---")  # Add a horizontal line for visual separation



    try:
        st.subheader("*Cluster Interpretation with SHAP*")
        st.caption("A SHAP summary plot shows the importance of different features for each cluster. The plot ranks the features based on their contribution to each cluster. Features with higher SHAP values on the plot have a stronger influence on that cluster. By looking at the plot, you can understand which features are more important in distinguishing and defining the characteristics of each cluster.")
        
        if dataset is not None:

            # Create a SHAP summary plot

            # Create a SHAP summary plot
            fig, ax = plt.subplots(figsize=(8, 5))  # Adjust the figsize as needed
            shap.summary_plot(shap_values, dataset)

            # Set the figure background color to transparent
            fig.patch.set_alpha(0.0)

            # Set the axes background color to transparent
            ax.patch.set_alpha(0.0)

            # Remove the border around the plot
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.spines["left"].set_visible(False)

            # Set the ticks and labels color to black
            ax.tick_params(axis='x', colors='black')
            ax.tick_params(axis='y', colors='black')

            # Set the axis labels color to black
            plt.xlabel('Shap Values', color='black')  # Set x-axis title

            # Make the x and y-axis lines bold
            plt.axhline(0, color='black', linewidth=0)  # Horizontal line
            plt.axvline(0, color='black', linewidth=0)  # Vertical line

            # Adjust legend properties and location
            legend = plt.legend(loc='lower right', frameon=False)  # Move the legend to the bottom right corner and remove box outline
            plt.setp(legend.get_texts(), color='black')  # Set the legend text color

            # Adjust the plot layout to fit well in the figure
            plt.tight_layout()

            # Save the modified plot as a transparent image
            fig.savefig('shap_summary_plot.png', transparent=True, bbox_inches='tight')

            # Display the plot using Streamlit
            st.image('shap_summary_plot.png')

    except Exception as e:
        st.error("Error in cluster Interpretation with SHAP")
 