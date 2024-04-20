import os
import pickle
import random
import argparse
import pandas as pd
from tqdm import tqdm
import networkx as nx

from config import Config
random.seed(42)


def build_graph(conf):
    df = pd.read_csv(os.path.join(conf.ROOT_PATH + 'train_set.csv'))[['userid', 'feedid', 'play']]
    df['play'] = df['play'].astype(float)
    df = df[df['play'] > 0]
    user_watch_time = df.groupby('userid')['play'].sum()
    feed_watch_time = df.groupby('feedid')['play'].sum()
    G = nx.DiGraph()
    for index, row in df.iterrows():
        userid = row['userid']
        feedid = row['feedid']
        G.add_node(userid, bipartite=0)
        G.add_node(feedid, bipartite=1)
        watch_time = row['play']
        user_normalized_watch_time = watch_time / user_watch_time[userid]
        feed_normalized_watch_time = watch_time / feed_watch_time[feedid]
        G.add_edge(userid, feedid, weight=user_normalized_watch_time)
        G.add_edge(feedid, userid, weight=feed_normalized_watch_time)
    return G


def target_domain_generation(G, H, R):
    selected_feeds = {}
    user_nodes = [node for node in G.nodes() if G.nodes[node]['bipartite'] == 0]
    for start_node in tqdm(user_nodes):
        user_feed_counts = {}
        for _ in range(H):
            current_node = start_node
            for step in range(R):
                neighbors = list(G.neighbors(current_node))
                weights = [G.edges[current_node, neighbor]['weight'] for neighbor in neighbors]
                next_node = random.choices(neighbors, weights=weights)[0]
                current_node = next_node
                if step % 2 == 0:
                    if current_node in user_feed_counts:
                        user_feed_counts[current_node] += 1
                    else:
                        user_feed_counts[current_node] = 1
        direct_feed_counts = {}
        avg_count = 0
        for feedid, count in user_feed_counts.items():
            if feedid in G.neighbors(start_node):
                direct_feed_counts[feedid] = count
        if direct_feed_counts:
            avg_count = sum(direct_feed_counts.values()) / len(direct_feed_counts)
        selected_feed_list = []
        for feedid, count in user_feed_counts.items():
            if feedid not in G.neighbors(start_node) and avg_count != 0 and count > avg_count:
                selected_feed_list.append(feedid)
        selected_feeds[start_node] = selected_feed_list
    return selected_feeds


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_name', type=str, default='WeChat')
    args = parser.parse_args()

    conf = Config(args.dataset_name)

    G = build_graph(conf)
    H, R = 50, 25
    selected_feeds = target_domain_generation(G, H, R)
    total_count = sum(len(feeds) for feeds in selected_feeds.values())
    print(total_count)
    filename = os.path.join(conf.ROOT_PATH, f'selected_feeds_H_{H}_R_{R}.pkl')
    with open(filename, 'wb') as f:
        pickle.dump(selected_feeds, f)

    with open(filename, 'rb') as f:
        selected_feeds = pickle.load(f)
    target_domain_samples = pd.DataFrame([(str(int(user)), str(int(feed))) for user, feeds in selected_feeds.items() for feed in feeds], columns=['userid', 'feedid'])
    target_domain_samples.to_csv(os.path.join(conf.ROOT_PATH, 'target_domain_samples.csv'), index=False)
    target_domain_samples = pd.read_csv(os.path.join(conf.ROOT_PATH + 'target_domain_samples.csv'))[['userid', 'feedid']]

    user_features = pd.read_csv(os.path.join(conf.ROOT_PATH, 'user_features.csv'))
    feed_features = pd.read_csv(os.path.join(conf.ROOT_PATH, 'feed_features.csv'))
    target_domain_samples = pd.merge(target_domain_samples, user_features, on='userid', how='left')
    target_domain_samples = pd.merge(target_domain_samples, feed_features, on='feedid', how='left')
    target_domain_samples = target_domain_samples.dropna()
    target_domain_samples.to_csv(os.path.join(conf.ROOT_PATH, 'target_domain_samples_train_set.csv'), index=False)