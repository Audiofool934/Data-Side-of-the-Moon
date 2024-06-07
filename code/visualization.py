import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.cm as cm

# 1. 原始数据的散点图
def plot_original_data(encoded_samples, selected_labels, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    filtered_samples = encoded_samples[encoded_samples['label'].isin(selected_labels)]
    sns.scatterplot(data=filtered_samples, x='Enc. Variable 0', y='Enc. Variable 1', 
                    hue=filtered_samples.label.astype(str), alpha=0.7, palette='rainbow', ax=ax)
    ax.set_title('Scatter Plot of Encoded Variables')
    
    if ax is None:
        plt.show()

# 2. t-SNE 转换后的散点图
def plot_tsne(encoded_samples, X, selected_labels, ax=None, figsize=(10, 7)):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    tsne = TSNE(n_components=2)
    tsne_results = tsne.fit_transform(X)
    
    tsne_df = pd.DataFrame(tsne_results, columns=['tsne-2d-one', 'tsne-2d-two'])
    tsne_df['label'] = encoded_samples['label'].astype(str)
    
    filtered_tsne_df = tsne_df[tsne_df['label'].isin(selected_labels)]
    
    sns.scatterplot(data=filtered_tsne_df, x='tsne-2d-one', y='tsne-2d-two', hue='label', palette='rainbow', alpha=0.7, ax=ax)
    
    ax.set_title('t-SNE Scatter Plot')
    ax.set_xlabel('tsne-2d-one')
    ax.set_ylabel('tsne-2d-two')
    
    if ax is None:
        plt.show()

# 3. PCA 转换后的散点图
def plot_pca(encoded_samples, X, selected_labels, ax=None, figsize=(10, 7)):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(X)
    pca_df = pd.DataFrame(pca_results, columns=['pca-2d-one', 'pca-2d-two'])
    pca_df['label'] = encoded_samples['label'].astype(str)
    filtered_pca_df = pca_df[pca_df['label'].isin(selected_labels)]
    
    sns.scatterplot(data=filtered_pca_df, x='pca-2d-one', y='pca-2d-two', hue='label', palette='rainbow', alpha=0.7, ax=ax)
    ax.set_title('PCA Scatter Plot')
    ax.set_xlabel('pca-2d-one')
    ax.set_ylabel('pca-2d-two')
    
    if ax is None:
        plt.show()

# 4. 综合显示所有图表
def plot_all(encoded_samples, X, selected_labels):
    fig, axs = plt.subplots(3, 1, figsize=(7, 14))

    plot_original_data(encoded_samples, selected_labels, ax=axs[0])
    plot_tsne(encoded_samples, X, selected_labels, ax=axs[1])
    plot_pca(encoded_samples, X, selected_labels, ax=axs[2])

    plt.tight_layout()
    plt.show()
    
####################################
####################################

def graph_kmeans_pca(df, cluster_label, n_pca, graph_2d=True, graph_3d=True):
    # 提取 PCA 组件和聚类标签
    pca_components = pd.DataFrame(df[f'PCA_{n_pca}'].tolist(), index=df.index)
    labels = df[cluster_label]

    if graph_2d and pca_components.shape[1] >= 2:
        plt.figure(figsize=(8, 6))
        plt.scatter(pca_components.iloc[:, 0], pca_components.iloc[:, 1], c=labels, cmap='viridis')
        plt.title('MFCC Clustering(2D)')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.colorbar()
        plt.show()

    if graph_3d and pca_components.shape[1] >= 3:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(pca_components.iloc[:, 0], pca_components.iloc[:, 1], pca_components.iloc[:, 2], c=labels,
                             cmap='viridis', s=100)
        ax.set_title('Song Clustering(3D)')
        ax.set_xlabel('PCA Component 1')
        ax.set_ylabel('PCA Component 2')
        ax.set_zlabel('PCA Component 3')
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
        cbar.set_label('Cluster Label')
        plt.show()

def graph_kmeans_tsne(df, cluster_label, graph_2d=True, graph_3d=True):
    # 提取 PCA 组件和聚类标签
    pca_components = pd.DataFrame(df['t-SNE'].tolist(), index=df.index)
    labels = df[cluster_label]

    if graph_2d and pca_components.shape[1] >= 2:
        plt.figure(figsize=(8, 6))
        plt.scatter(pca_components.iloc[:, 0], pca_components.iloc[:, 1], c=labels, cmap='viridis')
        plt.title('Song Clustering(2D)')
        plt.xlabel('tsne Component 1')
        plt.ylabel('tsne Component 2')
        plt.colorbar()
        plt.show()

    if graph_3d and pca_components.shape[1] >= 3:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(pca_components.iloc[:, 0], pca_components.iloc[:, 1], pca_components.iloc[:, 2], c=labels,
                             cmap='viridis', s=100)
        ax.set_title('Song Clustering(3D)')
        ax.set_xlabel('tsne Component 1')
        ax.set_ylabel('tsne Component 2')
        ax.set_zlabel('tsne Component 3')
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
        cbar.set_label('Cluster Label')
        plt.show()

def clusters_years(mfcc_df, cluster_label, normalize=True):
    mfcc_df['Year'] = mfcc_df['Year'].astype(int)

    # Verify unique years
    #print("Unique years:", mfcc_df['Year'].unique())

    # Prepare the data for plotting
    # Group by 'Year' and 'Cluster', and count the occurrences
    cluster_year_data = mfcc_df.groupby(['Year', cluster_label]).size().unstack(fill_value=0)
    num_clusters = len(mfcc_df[cluster_label].unique())
    cmap = cm.get_cmap('Spectral', num_clusters)
    colors = [cmap(i) for i in range(num_clusters)]

    if normalize:
        # Normalize the data to show proportion instead of count
        cluster_year_normalized = cluster_year_data.div(cluster_year_data.sum(axis=1), axis=0)
        # Plot a stacked bar chart with normalized values
        ax = cluster_year_normalized.plot(kind='bar', stacked=True, figsize=(10, 6), color=colors, width=0.8)
        plt.title('Cluster Distribution Over Years(Albums)')
        plt.xlabel('Year')
        plt.ylabel('Proportion of Tracks')
    else:
        # Plot a stacked bar chart with count values
        ax = cluster_year_data.plot(kind='bar', stacked=True, figsize=(10, 6), color=colors, width=0.8)
        plt.title('Cluster Distribution Over Years(Albums)')
        plt.xlabel('Year')
        plt.ylabel('Count of Tracks')

    # Set x-axis ticks to show each year
    ax.set_xticks(range(len(cluster_year_data.index)))
    ax.set_xticklabels(cluster_year_data.index, rotation=45)

    plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()