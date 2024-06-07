import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd

def kmeans_and_save(filepath, output_path, n_clusters, n_pca=3, n_tsne=3, Mode='AE'):

    df_extract = pd.read_hdf(filepath, key="extract")
    extract_matrix = np.vstack(df_extract[Mode].to_list())
    scaler = StandardScaler()
    extract_features_scaled = scaler.fit_transform(extract_matrix)

    # K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    clusters = kmeans.fit_predict(extract_features_scaled)
    df_extract[f'Cluster'] = clusters
    
    # PCA
    pca = PCA(n_components=n_pca, random_state=0)
    pca_features = pca.fit_transform(extract_features_scaled)
    df_extract[f'PCA_{n_pca}'] = list(pca_features)

    # t-SNE
    tsne = TSNE(n_components=n_tsne, random_state=0)
    tsne_features = tsne.fit_transform(extract_features_scaled)
    df_extract[f't-SNE_{n_tsne}'] = list(tsne_features)

    df_extract.to_hdf(output_path, key='extract', mode='w')

if __name__=="__main__":
    pass