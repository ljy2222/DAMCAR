import os
import torch
import pickle
import argparse
import pandas as pd

from config import Config
from model.dcn import DCN
from model.deepfm import DeepFM
from preprocessing.inputs import SparseFeat, DenseFeat, get_feature_names


def get_ranking_scores(conf, ranking_model_name):
    train = pd.read_csv(os.path.join(conf.ROOT_PATH + 'train_set.csv'))[['userid', 'feedid'] + conf.FEAT_LIST]
    dst_train = pd.read_csv(os.path.join(conf.ROOT_PATH + 'target_domain_samples_train_set.csv'))[
            ['userid', 'feedid'] + conf.sparse_features + conf.dense_features]
    train = pd.concat([train, dst_train])
    with open(os.path.join(conf.ROOT_PATH, 'feat_nunique.pkl'), 'rb') as file:
        feat_nunique = pickle.load(file)

    sparse_features_columns = [SparseFeat(feat, vocabulary_size=feat_nunique[feat], embedding_dim=conf.EMBED_DIMENSIONS) for feat in conf.sparse_features]
    dense_features_columns = [DenseFeat(feat, dimension=1) for feat in conf.dense_features]
    linear_feature_columns = sparse_features_columns + dense_features_columns
    dnn_feature_columns = sparse_features_columns + dense_features_columns
    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    train_model_input = {name: train[name] for name in feature_names}

    MODEL_PATH = os.path.join(conf.ROOT_PATH + f'model_{ranking_model_name}.pt')
    if ranking_model_name == 'deepfm':
        model = DeepFM(linear_feature_columns, dnn_feature_columns, task='binary', l2_reg_embedding=1e-5, device=conf.device)
    elif ranking_model_name == 'dcn':
        model = DCN(linear_feature_columns, dnn_feature_columns, task='binary', l2_reg_embedding=1e-5, device=conf.device)

    model.load_state_dict(torch.load(MODEL_PATH))
    pred_logits = model.predict(train_model_input, 128)
    train['label2'] = pred_logits
    train = train[['userid', 'feedid', 'label2']]
    train.to_csv(os.path.join(conf.ROOT_PATH, f'train_set_DAMCAR_{ranking_model_name}.csv'), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_name', type=str, default='WeChat')
    parser.add_argument('-r', '--ranking_model_name', type=str, default='deepfm')
    args = parser.parse_args()

    conf = Config(args.dataset_name)
    get_ranking_scores(conf, args.ranking_model_name)