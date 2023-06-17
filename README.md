# Insightful_Clusters_App

[My Streamlit app can be found here!](https://aswinram1997-insightful-clusters-app-app-ttcgl5.streamlit.app)

<p align="center">
  <img src="![Cluster Visualization](https://github.com/aswinram1997/Insightful_Clusters_App/assets/102771069/b95b1228-0f04-4990-b6c0-9e10cc57ef50)" alt="Cluster Visualization">
</p>

  
## Introduction
 [Insightful Clusters](https://aswinram1997-insightful-clusters-app-app-ttcgl5.streamlit.app) is a user-friendly application designed to provide valuable insights from datasets without requiring extensive data science expertise. Built using the Streamlit framework and deployed on Streamlit Sharing, this powerful app offers automated preprocessing, clustering, and interpretation tasks, making it accessible to users with varying levels of technical knowledge.

## Why Insightful Clusters?
Extracting meaningful insights from complex datasets can be challenging, especially for non-data scientists. Insightful Clusters addresses this need by offering a simplified solution that automates key processes, enabling users to uncover hidden patterns and gain valuable insights with ease. Insightful Clusters caters to a both technical and non-technical users. Insightful Clusters has diverse applications in fields like marketing, finance, healthcare, social sciences, and e-commerce. It enables customer segmentation in marketing, aids investment decision-making in finance, supports patient profiling in healthcare, assists social science research, and enhances customer analysis in e-commerce.

## Impact of Insightful Clusters
Insightful Clusters provides several impactful benefits to its users:

- Ease of Use: The user-friendly interface and automated processes make it accessible to users with varying technical expertise. Users can upload datasets, preprocess data, perform automatic or manual clustering, and interpret results without extensive data science knowledge.

- Time Efficiency: By automating preprocessing and clustering, Insightful Clusters saves users significant time and effort. Users can obtain meaningful insights quickly and focus on informed decision-making.

- Interpretability: The integration of the XGBoost model and SHAP summary plot enables users to understand the meaning of clusters and the key features driving results. This empowers users to gain deeper insights and make informed decisions based on cluster characteristics.

## Modules of Insightful Clusters
Insightful Clusters consists of the following modules:

- Preloaded Datasets and Custom Upload: Users can choose from preloaded datasets or upload their own. This flexibility allows exploration of the app's capabilities. Note: Long strings in datasets may not be accommodated in the current version.

- Data Preprocessing: This module handles missing values and performs feature extraction. Numerical features are normalized, and categorical features are encoded. It's advised to preprocess data beforehand for exceptionally unclean or specific preprocessing needs. Note: Long strings in datasets may not be accommodated.

- Automatic and Manual Clustering: Insightful Clusters offers automatic and manual clustering options. Automatic clustering applies advanced algorithms (K-means and Agglomerative Clustering) to identify patterns. Manual clustering provides more control over the process, allowing users to select algorithms and specify the number of clusters.

- XGBoost Model for Cluster Interpretation: Insightful Clusters employs an XGBoost classification model to interpret clusters. Users can understand the key features driving clustering results and gain deeper insights into their data. However, it must be acknowledged that due to the real-time nature of the training process, the model does not undergo fine-tuning of hyperparameters specifically aimed at reducing training time. 

- SHAP Summary Plot: Insightful Clusters provides a SHAP summary plot to aid in XGBoost model interpretation. This plot visualizes feature importance and contributions to each cluster. Users can analyze the plot to understand the impact of features on cluster formation.

## Conclusion
In conclusion, Insightful Clusters is a user-friendly and powerful application that simplifies data clustering and interpretation, providing valuable insights for users across various domains. Its impact is evident through its ease of use, time efficiency, and interpretability. With applications in marketing, finance, healthcare, social sciences, and e-commerce, Insightful Clusters empowers users to uncover hidden patterns, make informed decisions, and drive success in their respective fields. 

## Limitations and Future Directions
Insightful Clusters, while a powerful tool, has a few limitations that can guide its future development:

- Handling Long Strings: Current version may have limitations in processing datasets with long strings. Future updates could improve support for such datasets.

- Preprocessing Complex Data: Highly unclean or specialized datasets may require additional preprocessing steps beyond the app's automation. Enhancements could offer advanced preprocessing options for diverse data types.

- Expansion of Clustering Algorithms: Currently, Insightful Clusters supports K-means and Agglomerative Clustering. Future updates could include more clustering algorithms like DBSCAN or Gaussian Mixture Models to enhance analysis options.





