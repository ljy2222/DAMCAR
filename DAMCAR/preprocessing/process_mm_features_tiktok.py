import os
import pandas as pd
from sklearn.decomposition import PCA

from config import Config


def get_text_features(conf):
    title_feats = pd.read_json(conf.FEED_EMBEDDINGS_1, lines=True)
    def get_title_len(d):
        return sum(d.values())
    title_feats['title_len'] = title_feats['title_features'].apply(get_title_len)
    prior = title_feats['title_len'].mean()
    title_feats['title_features'] = title_feats['title_features'].apply(lambda x: list(x.keys()))
    title_feats['title_len'].fillna(prior, inplace=True)
    title_feats = title_feats.rename(columns={'item_id': 'feedid'})
    title_feats.to_csv(os.path.join(conf.ROOT_PATH, 'processed_text_features.csv'), index=False)


def get_video_features(conf):
    feed_video_embed = pd.read_json(conf.FEED_EMBEDDINGS_2, lines=True)
    feed_video_embed_df = pd.DataFrame(feed_video_embed.video_feature_dim_128.tolist(), columns=['vd' + str(i) for i in range(128)])
    feed_video_embed_df.fillna(0, inplace=True)
    pca = PCA(n_components=conf.PCA_DIMENSIONS, random_state=conf.SEED)
    pca_embed_df = pd.DataFrame(pca.fit_transform(feed_video_embed_df))
    del feed_video_embed['video_feature_dim_128']
    pca_embed_df = pd.concat([feed_video_embed, pca_embed_df], axis=1)
    pca_embed_df = pca_embed_df.rename(columns={'item_id': 'feedid'})
    for i in range(conf.PCA_DIMENSIONS):
        pca_embed_df = pca_embed_df.rename(columns={i: 'vd{}'.format(i)})
    pca_embed_df.to_csv(os.path.join(conf.ROOT_PATH, 'pca_video_embeddings.csv'), index=False)


def get_audio_features(conf):
    feed_audio_embed = pd.read_json(conf.FEED_EMBEDDINGS_3, lines=True)
    feed_audio_embed_df = pd.DataFrame(feed_audio_embed.audio_feature_128_dim.tolist(), columns=['ad' + str(i) for i in range(128)])
    feed_audio_embed_df.fillna(0, inplace=True)
    pca = PCA(n_components=conf.PCA_DIMENSIONS, random_state=conf.SEED)
    pca_embed_df = pd.DataFrame(pca.fit_transform(feed_audio_embed_df))
    del feed_audio_embed['audio_feature_128_dim']
    pca_embed_df = pd.concat([feed_audio_embed, pca_embed_df], axis=1)
    pca_embed_df = pca_embed_df.rename(columns={'item_id': 'feedid'})
    for i in range(conf.PCA_DIMENSIONS):
        pca_embed_df = pca_embed_df.rename(columns={i: 'ad{}'.format(i)})
    pca_embed_df.to_csv(os.path.join(conf.ROOT_PATH, 'pca_audio_embeddings.csv'), index=False)


if __name__ == '__main__':
    dataset_name = 'TikTok'
    conf = Config(dataset_name)
    get_text_features(conf)
    get_video_features(conf)
    get_audio_features(conf)