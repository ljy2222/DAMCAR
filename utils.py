import numpy as np
from sklearn.metrics import log_loss, roc_auc_score


def cal_logloss_auc(y_true, y_pred):
    auc = roc_auc_score(y_true, y_pred)
    logloss = log_loss(y_true, y_pred)
    auc, logloss = round(auc, 4), round(logloss, 4)
    return auc, logloss


def cal_ndcg_map_hr(y_true_dict, y_pred_dict, K):
    ndcg_list = []
    map_list = []
    hr_list = []
    for uid in y_true_dict.keys():
        y_pred = [0 for _ in range(K)]
        for index, item in enumerate(y_pred_dict[uid]):
            if item in y_true_dict[uid]:
                y_pred[index] = 1

        dcg = np.sum(y_pred / np.log2(np.arange(2, len(y_pred) + 2)))
        y_true = sorted(y_pred, reverse=True)
        idcg = np.sum(y_true / np.log2(np.arange(2, len(y_pred) + 2)))
        ndcg = dcg / idcg if idcg != 0 else 0
        ndcg_list.append(ndcg)

        sum = 0
        hits = 0
        for n in range(len(y_pred)):
            if y_pred[n] == 1:
                hits += 1
                sum += hits / (n + 1.0)
        map = sum / hits if hits != 0 else 0
        map_list.append(map)
        hr_list.append(hits / K)

    ndcg, map, hr = round(np.mean(ndcg_list), 4), round(np.mean(map_list), 4), round(np.mean(hr_list), 4)
    return ndcg, map, hr