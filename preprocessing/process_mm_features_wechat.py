import os
import pandas as pd
from sklearn.decomposition import PCA

from config import Config


def process(x):
    num_list = x.split(' ')[:-1]
    res = {}
    for i, num in enumerate(num_list):
        res[i] = float(num)
    return pd.Series(res)


def process_embed(embed_df, PCA_DIMENSIONS, SEED):
    new_embed_df = embed_df.feed_embedding.apply(process)
    pca = PCA(n_components=PCA_DIMENSIONS, random_state=SEED)
    new_embed_df = pd.DataFrame(pca.fit_transform(new_embed_df))
    del embed_df['feed_embedding']
    new_embed_df = pd.concat([embed_df, new_embed_df], axis=1)
    return new_embed_df


if __name__ == '__main__':
    dataset_name = 'WeChat'
    conf = Config(dataset_name)
    feed_embed_df = pd.read_csv(conf.FEED_EMBEDDINGS)
    pca_embed_df = process_embed(feed_embed_df, conf.PCA_DIMENSIONS, conf.SEED)
    pca_embed_df.to_csv(os.path.join(conf.ROOT_PATH, 'pca_embeddings.csv'), index=False)