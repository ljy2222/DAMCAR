import os
import torch
import pickle
import argparse
import pandas as pd

from utils import *
from config import Config
from model.dcn import DCN
from model.deepfm import DeepFM
from preprocessing.inputs import SparseFeat, DenseFeat, get_feature_names


def train_ranking_model(conf, ranking_model_name):
    train = pd.read_csv(os.path.join(conf.ROOT_PATH + 'train_set.csv'))[['userid', 'feedid'] + conf.FEAT_LIST]
    valid = pd.read_csv(os.path.join(conf.ROOT_PATH + 'valid_set.csv'))[['userid', 'feedid'] + conf.FEAT_LIST]
    test = pd.read_csv(os.path.join(conf.ROOT_PATH + 'test_set.csv'))[['userid', 'feedid'] + conf.FEAT_LIST]
    with open(os.path.join(conf.ROOT_PATH, 'feat_nunique.pkl'), 'rb') as file:
        feat_nunique = pickle.load(file)

    sparse_features_columns = [SparseFeat(feat, vocabulary_size=feat_nunique[feat], embedding_dim=conf.EMBED_DIMENSIONS) for feat in conf.sparse_features]
    dense_features_columns = [DenseFeat(feat, dimension=1) for feat in conf.dense_features]
    linear_feature_columns = sparse_features_columns + dense_features_columns
    dnn_feature_columns = sparse_features_columns + dense_features_columns
    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    train_model_input = {name: train[name] for name in feature_names}
    valid_model_input = {name: valid[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}

    MODEL_PATH = os.path.join(conf.ROOT_PATH + f'model_{ranking_model_name}.pt')
    if ranking_model_name == 'deepfm':
        model = DeepFM(linear_feature_columns, dnn_feature_columns, task='binary', l2_reg_embedding=1e-5, device=conf.device)
    elif ranking_model_name == 'dcn':
        model = DCN(linear_feature_columns, dnn_feature_columns, task='binary', l2_reg_embedding=1e-5, device=conf.device)
    model.compile('adagrad', 'binary_crossentropy', metrics=['auc', 'binary_crossentropy'])
    model.fit(train_model_input, train[conf.target].values, batch_size=512, epochs=3, verbose=1, validation_data=[valid_model_input, valid[conf.target].values])
    torch.save(model.state_dict(), MODEL_PATH)

    model.load_state_dict(torch.load(MODEL_PATH))
    true_labels = test[conf.target].values
    pred_logits = model.predict(test_model_input, 128)
    auc, logloss = cal_logloss_auc(true_labels, pred_logits)
    print(f'auc: {auc}, logloss: {logloss}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_name', type=str, default='WeChat')
    parser.add_argument('-r', '--ranking_model_name', type=str, default='deepfm')
    args = parser.parse_args()

    conf = Config(args.dataset_name)
    train_ranking_model(conf, args.ranking_model_name)