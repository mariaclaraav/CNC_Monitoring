from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, scale
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score, silhouette_score
from sklearn.datasets import make_blobs
import umap


def plot_pca_cumulative_variance(X_scaled, n_components=None):
    """
    Fit a PCA model to the given scaled data and plot the cumulative explained variance.

    Parameters:
    - X_scaled: The scaled input data (e.g., a NumPy array or DataFrame).
    - n_components: Number of PCA components to consider (default is None, which means all).
    """
    # Initialize PCA model
    pca = PCA(n_components=n_components)
    
    # Fit PCA model to the scaled data
    pca.fit(X_scaled)
    
    # Calculate the cumulative explained variance
    cumulative_variance = pca.explained_variance_ratio_.cumsum()
    
    # Create a plot of the cumulative explained variance
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--', color='b')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA - Cumulative Variance Explained by Components')
    plt.grid(True)
    plt.show()

def fit_gmm_evaluate(df, columns, n_components_range, random_state=0,covariance_type = 'diag'):
    """
    Fit Gaussian Mixture Models for a range of component numbers and evaluate using several metrics.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        columns (list): List of column names to be used in the model.
        n_components_range (range): Range of components numbers to fit the models for.
        random_state (int): Random state for reproducibility of the models.
        
    Returns:
        dict: Dictionary containing fitted models and evaluation metrics.
    """
    # Prepare the data
    df_copy = df[columns].copy()
    data = df_copy
    # Standardizing the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)
    data= pd.DataFrame(X_scaled, columns=columns)
    
    # Storage for models and metrics
    models = []
    bics = []
    log_likelihoods = []
    davies_bouldin_indices = []
    calinski_harabasz_indices = []
    
    # Fit models and compute metrics
    for n in tqdm(n_components_range, desc='Fitting Models'):
        gmm = GaussianMixture(n_components=n, covariance_type=covariance_type, random_state=random_state,
                              verbose=1, verbose_interval=10).fit(data)
        models.append(gmm)
        bics.append(gmm.bic(data))
        log_likelihoods.append(gmm.score(data) * len(data))  # Adjusted log likelihood
        
        # Predict the labels
        labels = gmm.predict(data)
        
        # Calculate metrics if there is more than one cluster
        if n > 1:
            davies_bouldin_indices.append(davies_bouldin_score(data, labels))
            calinski_harabasz_indices.append(calinski_harabasz_score(data, labels))
        else:
            davies_bouldin_indices.append(None)
            calinski_harabasz_indices.append(None)
    
    return {
        "models": models,
        "bics": bics,
        "log_likelihoods": log_likelihoods,
        "davies_bouldin_indices": davies_bouldin_indices,
        "calinski_harabasz_indices": calinski_harabasz_indices
    }