import os
import torch
import pickle
import argparse
import pandas as pd

from config import Config
from model.ada import ADA
from model.dssm import DSSM
from evaluation import test_main
from preprocessing.inputs import SparseFeat, DenseFeat, get_feature_names


def DAMCAR_training(conf, debias_method, ranking_model_name, alpha, lambda1, lambda2, lambda3):
    src_train = pd.read_csv(os.path.join(conf.ROOT_PATH + 'train_set.csv'))[['userid', 'feedid'] + conf.FEAT_LIST]
    dst_train = pd.read_csv(os.path.join(conf.ROOT_PATH + 'target_domain_samples_train_set.csv'))[['userid', 'feedid'] + conf.sparse_features + conf.dense_features]
    src_train_1 = src_train[src_train['label'] == 1]
    src_train_2 = src_train[src_train['label'] == 0]
    train = pd.concat([src_train_2, dst_train])
    label2 = pd.read_csv(os.path.join(conf.ROOT_PATH, f'train_set_{debias_method}_{ranking_model_name}.csv'))
    valid = pd.read_csv(os.path.join(conf.ROOT_PATH + 'valid_set.csv'))[['userid', 'feedid'] + conf.FEAT_LIST]
    with open(os.path.join(conf.ROOT_PATH, 'feat_nunique.pkl'), 'rb') as file:
        feat_nunique = pickle.load(file)

    sparse_features_columns = [SparseFeat(feat, vocabulary_size=feat_nunique[feat], embedding_dim=conf.EMBED_DIMENSIONS) for feat in conf.sparse_features]
    dense_features_columns = [DenseFeat(feat, dimension=1) for feat in conf.dense_features]
    linear_feature_columns = sparse_features_columns + dense_features_columns
    dnn_feature_columns = sparse_features_columns + dense_features_columns
    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    # copy for training
    while len(dst_train) < len(src_train):
        dst_train = pd.concat([dst_train, dst_train]).reset_index(drop=True)
    dst_train = dst_train.iloc[:len(src_train)]
    assert len(dst_train) == len(src_train)

    src_train_model_input = {name: src_train[name] for name in feature_names}
    dst_train_model_input = {name: dst_train[name] for name in feature_names}
    train_model_input = {name: train[name] for name in feature_names}
    valid_model_input = {name: valid[name] for name in feature_names}

    MODEL_PATH = os.path.join(conf.ROOT_PATH + f'model_ada.pt')
    model = ADA(linear_feature_columns, dnn_feature_columns, task='binary', l2_reg_embedding=1e-5, device=conf.device)
    teacher_model = ADA(linear_feature_columns, dnn_feature_columns, task='binary', l2_reg_embedding=1e-5, device=conf.device)
    teacher_model.eval()
    model.compile('adagrad', 'binary_crossentropy', metrics=['auc', 'binary_crossentropy'], alpha=alpha, lambda1=lambda1, lambda2=lambda2)
    model.fit(src_train_model_input, src_train['label'].values, dst_train_model_input, batch_size=512, epochs=3, verbose=1,
              validation_data=[valid_model_input, valid[conf.target].values], teacher_model=teacher_model)
    torch.save(model.state_dict(), MODEL_PATH)
    pred_logits = model.predict(train_model_input, 128)
    train['label'] = pred_logits
    train = pd.concat([src_train_1, train])
    train = pd.merge(train, label2, on=['userid', 'feedid'], how='left')
    train_model_input = {name: train[name] for name in feature_names}

    user_features_columns = [SparseFeat(feat, vocabulary_size=feat_nunique[feat], embedding_dim=conf.EMBED_DIMENSIONS) for feat in conf.user_sparse_features]
    item_features_columns = ([SparseFeat(feat, vocabulary_size=feat_nunique[feat], embedding_dim=conf.EMBED_DIMENSIONS) for feat in conf.item_sparse_features] +
                             [DenseFeat(feat, dimension=1) for feat in conf.item_dense_features])
    MODEL_PATH = os.path.join(conf.ROOT_PATH + f'model_dssm_{debias_method}_{ranking_model_name}.pt')
    model = DSSM(user_features_columns, item_features_columns, task='binary', l2_reg_embedding=1e-5, device=conf.device)
    model.compile('adagrad', 'binary_crossentropy', metrics=['auc', 'binary_crossentropy'], lambda3=lambda3)
    model.fit(train_model_input, train['label'].values, y2=train['label2'].values, batch_size=512, epochs=3, verbose=1,
              validation_data=[valid_model_input, valid[conf.target].values])
    torch.save(model.state_dict(), MODEL_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_name', type=str, default='WeChat')
    parser.add_argument('-m', '--debias_method', type=str, default='DAMCAR')
    parser.add_argument('-r', '--ranking_model_name', type=str, default='deepfm')
    parser.add_argument('--alpha', type=float, default=0.99)
    parser.add_argument('--lambda1', type=float, default=0.5)
    parser.add_argument('--lambda2', type=float, default=0.2)
    parser.add_argument('--lambda3', type=float, default=0.3)
    args = parser.parse_args()

    conf = Config(args.dataset_name)
    DAMCAR_training(conf, args.debias_method, args.ranking_model_name, args.alpha, args.lambda1, args.lambda2, args.lambda3)
    test_main(conf, args.debias_method, args.ranking_model_name)