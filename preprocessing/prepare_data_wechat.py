import os
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from config import Config


def prepare_data(conf):
    user_action_df = pd.read_csv(conf.USER_ACTION)
    user_action_df = user_action_df.sort_values(by=['date_'])
    user_action_df = user_action_df.drop_duplicates(['userid', 'feedid'], keep='last')
    feed_info_df = pd.read_csv(conf.FEED_INFO)

    interactions = user_action_df[['userid', 'feedid', 'date_']].drop_duplicates(['userid', 'feedid'], keep='last')
    dataset = pd.merge(interactions, user_action_df[['userid', 'feedid', 'play']], on=['userid', 'feedid'], how='left')
    dataset = pd.merge(dataset, feed_info_df[['feedid', 'videoplayseconds']], on='feedid', how='left')
    dataset['label'] = dataset.apply(lambda row: 1 if row['play'] >= (row['videoplayseconds'] * 1000) else 0, axis=1)
    dataset = dataset.drop_duplicates(['userid', 'feedid'], keep='last')
    dataset = dataset.sort_values(by=['date_'])
    dataset = dataset[['userid', 'feedid', 'label', 'play']]
    dataset.to_csv(os.path.join(conf.ROOT_PATH, 'userid_feedid_label.csv'), index=False)

    user_features = dataset[['userid']].drop_duplicates(['userid'], keep='last')
    user_features[conf.user_sparse_features] = user_features[['userid']]
    for feat in conf.user_sparse_features:
        lbe = LabelEncoder()
        user_features[feat] = lbe.fit_transform(user_features[feat])
    user_features.to_csv(os.path.join(conf.ROOT_PATH, 'user_features.csv'), index=False)

    feed_features = dataset[['feedid']].drop_duplicates(['feedid'], keep='last')
    feed_features[[conf.item_sparse_features[0]]] = feed_features[['feedid']]
    feed_features = pd.merge(feed_features, feed_info_df[conf.FEAT_FEED_LIST], on='feedid', how='left')
    pca_embed_df = pd.read_csv(os.path.join(conf.ROOT_PATH + 'pca_embeddings.csv'))
    feed_features = pd.merge(feed_features, pca_embed_df, on='feedid', how='left')
    feed_features[conf.item_sparse_features] = feed_features[conf.item_sparse_features].fillna(0)
    feed_features[conf.item_dense_features] = feed_features[conf.item_dense_features].fillna(0)
    for feat in conf.item_sparse_features:
        lbe = LabelEncoder()
        feed_features[feat] = lbe.fit_transform(feed_features[feat])
    for feat in conf.item_dense_features:
        mms = MinMaxScaler(feature_range=(0, 1))
        feed_features[[feat]] = mms.fit_transform(feed_features[[feat]])
    feed_features.to_csv(os.path.join(conf.ROOT_PATH, 'feed_features.csv'), index=False)

    dataset = pd.merge(dataset, user_features, on='userid', how='left')
    dataset = pd.merge(dataset, feed_features, on='feedid', how='left')
    dataset.to_csv(os.path.join(conf.ROOT_PATH, 'dataset.csv'), index=False)

    dataset = pd.read_csv(os.path.join(conf.ROOT_PATH + 'dataset.csv'))
    train = dataset.groupby('userid').apply(lambda x: x[:int(len(x) * 0.6)]).reset_index(drop=True)
    valid = dataset.groupby('userid').apply(lambda x: x[int(len(x) * 0.6):int(len(x) * 0.8)]).reset_index(drop=True)
    test = dataset.groupby('userid').apply(lambda x: x[int(len(x) * 0.8):]).reset_index(drop=True)

    ground_truth_list = test[test['label'] == 1].groupby('userid')['feedid'].agg(list).reset_index(name='feedid_list')
    test_true_label = {}
    for index, row in ground_truth_list.iterrows():
        test_true_label[row['userid']] = row['feedid_list']
    with open(os.path.join(conf.ROOT_PATH, 'test_true_label.pkl'), 'wb') as file:
        pickle.dump(test_true_label, file)

    dataset = pd.read_csv(os.path.join(conf.ROOT_PATH + 'dataset.csv'))[['userid', 'feedid'] + conf.FEAT_LIST]
    feat_nunique = {}
    for feat in conf.sparse_features:
        feat_nunique[feat] = dataset[feat].nunique()
    with open(os.path.join(conf.ROOT_PATH, 'feat_nunique.pkl'), 'wb') as file:
        pickle.dump(feat_nunique, file)

    train.to_csv(os.path.join(conf.ROOT_PATH, 'train_set.csv'), index=False)
    valid.to_csv(os.path.join(conf.ROOT_PATH, 'valid_set.csv'), index=False)
    test.to_csv(os.path.join(conf.ROOT_PATH, 'test_set.csv'), index=False)


if __name__ == '__main__':
    dataset_name = 'WeChat'
    conf = Config(dataset_name)
    prepare_data(conf)