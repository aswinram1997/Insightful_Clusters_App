from sklearn.cluster import KMeans, AgglomerativeClustering, OPTICS
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import numpy as np

def define_k_values_range():
    """
    Defines the range of k values for clustering.

    Returns:
    - k_values_range (range): The range of k values.
    """
    k_values_range = range(3, 7)
    return k_values_range

def kmeans_clustering(dataset, k):
    """
    Performs K-means clustering on the dataset.

    Parameters:
    - dataset (pd.DataFrame): The dataset to be clustered.
    - k (int): The number of clusters.

    Returns:
    - labels (np.ndarray): The cluster labels assigned by K-means.
    """
    model = KMeans(n_clusters=k, n_init=10)
    labels = model.fit_predict(dataset)
    return labels

def agglomerative_clustering(dataset, k):
    """
    Performs Agglomerative clustering on the dataset.

    Parameters:
    - dataset (pd.DataFrame): The dataset to be clustered.
    - k (int): The number of clusters.

    Returns:
    - labels (np.ndarray): The cluster labels assigned by Agglomerative clustering.
    """
    model = AgglomerativeClustering(n_clusters=k)
    labels = model.fit_predict(dataset)
    return labels

def evaluate_clustering(dataset, labels):
    """
    Evaluates the clustering results using Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Index.

    Parameters:
    - dataset (pd.DataFrame): The dataset used for clustering.
    - labels (np.ndarray): The cluster labels assigned to the dataset.

    Returns:
    - silhouette (float): The Silhouette Score.
    - db (float): The Davies-Bouldin Index.
    - ch (float): The Calinski-Harabasz Index.
    """
    silhouette = round(silhouette_score(dataset, labels), 3)
    db = round(davies_bouldin_score(dataset, labels), 3)
    ch = round(calinski_harabasz_score(dataset, labels), 3)
    return silhouette, db, ch

def perform_clustering_evaluation(dataset, k_values_range):
    """
    Performs clustering and evaluation for different algorithms and k values.

    Parameters:
    - dataset (pd.DataFrame): The dataset to be clustered.
    - k_values_range (range): The range of k values.

    Returns:
    - evaluation_scores (dict): Dictionary containing the evaluation scores for each algorithm and k value combination.
    """
    evaluation_scores = {}
    algorithms = {
        'KMeans': kmeans_clustering,
        'Agglomerative': agglomerative_clustering
    }    
    for algorithm in algorithms:
        for k in k_values_range:
            labels = algorithms[algorithm](dataset, k)
            silhouette, db, ch = evaluate_clustering(dataset, labels)
            evaluation_scores[(algorithm, k)] = {'silhouette': silhouette, 'db': db, 'ch': ch}

    return evaluation_scores

def count_votes(evaluation_scores):
    """
    Counts the votes for each combination based on evaluation scores.

    Parameters:
    - evaluation_scores (dict): Dictionary containing the evaluation scores for each algorithm and k value combination.

    Returns:
    - votes (dict): Dictionary containing the vote counts for each combination.
    """
    votes = {}
    for combination in evaluation_scores:
        algorithm, k = combination
        silhouette = evaluation_scores[combination]['silhouette']
        db = evaluation_scores[combination]['db']
        ch = evaluation_scores[combination]['ch']
        if (algorithm, k) not in votes:
            votes[(algorithm, k)] = 0
        if silhouette >= max(evaluation_scores[x]['silhouette'] for x in votes) and \
           db <= min(evaluation_scores[x]['db'] for x in votes) and \
           ch >= max(evaluation_scores[x]['ch'] for x in votes):
            votes[(algorithm, k)] += 1
    return votes

def select_best_combination(votes, evaluation_scores):
    """
    Selects the best combination based on the votes and evaluation scores.

    Parameters:
    - votes (dict): Dictionary containing the vote counts for each combination.
    - evaluation_scores (dict): Dictionary containing the evaluation scores for each algorithm and k value combination.

    Returns:
    - best_combination (tuple): The best combination of algorithm and k value.
    """
    max_votes = max(votes.values())
    winning_combinations = [combination for combination, count in votes.items() if count == max_votes]
    best_combination = max(winning_combinations, key=lambda x: (evaluation_scores[x]['silhouette'], -evaluation_scores[x]['db'], evaluation_scores[x]['ch']))
    return best_combination

def perform_final_clustering(dataset, algorithm, k):
    """
    Performs the final clustering with the best algorithm and k value combination.

    Parameters:
    - dataset (pd.DataFrame): The dataset to be clustered.
    - algorithm (str): The selected clustering algorithm.
    - k (int): The selected number of clusters.

    Returns:
    - labels (np.ndarray): The cluster labels assigned by the selected clustering algorithm.
    """
    if algorithm == 'KMeans':
        labels = kmeans_clustering(dataset, k)
    elif algorithm == 'Agglomerative':
        labels = agglomerative_clustering(dataset, k)
    elif algorithm == 'OPTICS':
        labels = agglomerative_clustering(dataset)
    return labels
