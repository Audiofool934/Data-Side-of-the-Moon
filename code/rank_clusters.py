import pandas as pd
import numpy as np

def generate_cluster_info(df, Mode='MFCC'):
    cluster_dict = {}
    # 获取唯一的聚类标签
    unique_clusters = df['Cluster'].unique()

    for cluster in unique_clusters:
        # 获取属于该聚类的所有行
        cluster_rows = df[df['Cluster'] == cluster]
        # 获取歌曲名称和年份
        songs_and_years = list(zip(cluster_rows['Song'], cluster_rows['Year']))
        # 计算方差
        values = np.vstack(cluster_rows[Mode].values)
        variance = np.var(values, axis=0).mean()
        # 计算歌曲数量
        song_count = len(cluster_rows)
        # 存储在字典中
        album_value = np.vstack(cluster_rows['Album'].values)
        cluster_dict[cluster] = {
            'Songs and Years': songs_and_years,
            'Variance': variance,
            'Song Count': song_count,
            'Album': album_value
        }
    order = list(sorted(cluster_dict.keys(), key=lambda x: cluster_dict[x]['Variance'], reverse=False))

    return cluster_dict, order

def show(df, output_file, Mode='MFCC'):
    cluster_dict, order = generate_cluster_info(df, Mode)
    with open(output_file, 'w') as f:
        for cluster in order:
            f.write(f"Cluster {cluster} [var:{cluster_dict[cluster]['Variance']:.3f}]\n(num:{cluster_dict[cluster]['Song Count']})\n")
            for song in cluster_dict[cluster]['Songs and Years']:
                f.write(f"{song[0]}---({song[1]})\n")
            f.write('---------\n\n')

def play_list(df, Mode='MFCC'):
    cluster_dict, order = generate_cluster_info(df, Mode)
    for cluster in order:
        if cluster_dict[cluster]['Song Count'] > 5:
            print('为你推荐的歌单:')
            for song, album in zip(cluster_dict[cluster]['Songs and Years'], cluster_dict[cluster]['Album']):
                print(f"{album[0]}: {song[0]}")
            break
