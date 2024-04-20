import os
import torch
import faiss
import pickle
import pandas as pd

from utils import *
from model.dcn import DCN
from model.dssm import DSSM
from model.deepfm import DeepFM
from preprocessing.inputs import SparseFeat, DenseFeat, get_feature_names


def test_main(conf, debias_method, ranking_model_name):
    with open(os.path.join(conf.ROOT_PATH, 'feat_nunique.pkl'), 'rb') as file:
        feat_nunique = pickle.load(file)

    user_features_columns = [SparseFeat(feat, vocabulary_size=feat_nunique[feat], embedding_dim=conf.EMBED_DIMENSIONS) for feat in conf.user_sparse_features]
    item_features_columns = [SparseFeat(feat, vocabulary_size=feat_nunique[feat], embedding_dim=conf.EMBED_DIMENSIONS)
                             for feat in conf.item_sparse_features] + [DenseFeat(feat, dimension=1) for feat in conf.item_dense_features]

    sparse_features_columns = [SparseFeat(feat, vocabulary_size=feat_nunique[feat], embedding_dim=conf.EMBED_DIMENSIONS) for feat in conf.sparse_features]
    dense_features_columns = [DenseFeat(feat, dimension=1) for feat in conf.dense_features]
    linear_feature_columns = sparse_features_columns + dense_features_columns
    dnn_feature_columns = sparse_features_columns + dense_features_columns
    feature_names_2 = get_feature_names(linear_feature_columns + dnn_feature_columns)

    test_set = pd.read_csv(os.path.join(conf.ROOT_PATH + 'test_set.csv'))[['userid', 'feedid'] + conf.FEAT_LIST]
    ground_truth_list = test_set[test_set['label'] == 1].groupby('userid')['feedid'].agg(list).reset_index(name='feedid_list')
    with open(os.path.join(conf.ROOT_PATH, 'test_true_label.pkl'), 'rb') as file:
        test_true_label = pickle.load(file)

    retrieval_model = DSSM(user_features_columns, item_features_columns, task='binary', l2_reg_embedding=1e-5, device=conf.device)
    MODEL_PATH = os.path.join(conf.ROOT_PATH + f'model_dssm_{debias_method}_{ranking_model_name}.pt')
    retrieval_model.load_state_dict(torch.load(MODEL_PATH))

    if ranking_model_name == 'deepfm':
        ranking_model = DeepFM(linear_feature_columns, dnn_feature_columns, task='binary', l2_reg_embedding=1e-5, device=conf.device)
    elif ranking_model_name == 'dcn':
        ranking_model = DCN(linear_feature_columns, dnn_feature_columns, task='binary', l2_reg_embedding=1e-5, device=conf.device)
    MODEL_PATH = os.path.join(conf.ROOT_PATH + f'model_{ranking_model_name}.pt')
    ranking_model.load_state_dict(torch.load(MODEL_PATH))

    dict_trained = retrieval_model.state_dict()
    user_embedding_model = DSSM(user_features_columns, [], task='binary', device=conf.device)
    dict_user = user_embedding_model.state_dict()
    for key in dict_user:
        dict_user[key] = dict_trained[key]
    user_embedding_model.load_state_dict(dict_user)
    user_profile = test_set[['userid'] + conf.user_sparse_features].drop_duplicates('userid')
    user_model_input = {name: user_profile[name] for name in conf.user_sparse_features}
    user_embs = user_embedding_model.predict(user_model_input, batch_size=2000)
    user_embs = user_embs.astype('float32')

    item_embedding_model = DSSM([], item_features_columns, task='binary', device=conf.device)
    dict_item = item_embedding_model.state_dict()
    for key in dict_item:
        dict_item[key] = dict_trained[key]
    item_embedding_model.load_state_dict(dict_item)
    item_profile = test_set[['feedid'] + conf.item_sparse_features + conf.item_dense_features].drop_duplicates('feedid')
    item_model_input = {name: item_profile[name] for name in conf.item_sparse_features + conf.item_dense_features}
    item_embs = item_embedding_model.predict(item_model_input, batch_size=2000)
    item_embs = item_embs.astype('float32')

    for K1 in [100, 200]:
        index = faiss.IndexFlatIP(user_embs.shape[1])
        index.add(item_embs)
        D, I = index.search(user_embs, K1)
        ranking_candidates = {}
        recalls, precisions = 0, 0
        for i, uid in enumerate(ground_truth_list['userid']):
            pred = [item_profile['feedid'].values[x] for x in I[i]]
            ranking_candidates[uid] = pred
            hit = len(set(pred[:K1]) & set(test_true_label[uid])) * 1.0
            recalls += hit / len(test_true_label[uid])
            precisions += hit / K1
        recall = round(recalls / len(ground_truth_list), 4)
        precision = round(precisions / len(ground_truth_list), 4)
        f1 = round(2 * (recall * precision) / (recall + precision), 4)
        print(f'recall@{K1}: {recall}, precision@{K1}: {precision}, f1@{K1}: {f1}')

        user_features = pd.read_csv(os.path.join(conf.ROOT_PATH, 'user_features.csv'))
        feed_features = pd.read_csv(os.path.join(conf.ROOT_PATH, 'feed_features.csv'))
        test = pd.DataFrame([(userid, feedid) for userid, feedids in ranking_candidates.items() for feedid in feedids],
                            columns=['userid', 'feedid'])
        test = pd.merge(test, user_features, on='userid', how='left')
        test = pd.merge(test, feed_features, on='feedid', how='left')
        test_model_input = {name: test[name] for name in feature_names_2}
        pred_logits = ranking_model.predict(test_model_input, 128)
        test['score'] = pd.DataFrame(pred_logits, columns=['score'])
        test = test.sort_values(by='score', ascending=False).reset_index(drop=True)

        K2 = int(K1 / 10)
        new_df = test.groupby('userid').head(K2)
        grouped = new_df.groupby('userid')['feedid'].agg(list).reset_index(name='feedid_list')
        grouped = grouped.groupby('userid')['feedid_list'].apply(list).to_dict()
        result_dict = {key: value[0] for key, value in grouped.items()}
        ndcg, map, hr = cal_ndcg_map_hr(test_true_label, result_dict, K2)
        print(f'ndcg@{K2}: {ndcg}, map@{K2}: {map}, hr@{K2}: {hr}')