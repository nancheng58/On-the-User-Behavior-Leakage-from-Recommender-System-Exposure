
import numpy as np


def metric_at_k_set_point(actual, predicted, topk, tgt_len):
    MRR = 0
    NDCG = 0
    Recall = 0
    num_users = len(predicted)
    total_user = 0
    for i in range(num_users):
        act_set = set(actual[i])
        pred_list = predicted[i]
        pred_set = set()
        total_user += 1

        for j in range(len(actual[i])):  # tgt_len is number of history data
            pred_position = pred_list[:topk]
            rank = -1
            truth = actual[i][j]
            for k in range(len(pred_position)):
                if pred_position[k] == truth:
                    rank = k
                    break
            if rank != -1:
                MRR += 1.0 / (rank + 1.0)
                NDCG += 1.0 / np.log2(rank + 2.0)
            pred_set = pred_set | set(pred_position)

        Recall += len(act_set & pred_set) / float(len(act_set))
    MRR /= tgt_len
    NDCG /= tgt_len
    return NDCG / total_user, MRR / total_user, Recall / total_user


def metric_at_k_set(actual, predicted, topk, tgt_len):
    MRR = 0
    NDCG = 0
    Recall = 0
    num_users = len(predicted)
    total_user = 0
    for i in range(num_users):
        act_set = set(actual[i])
        pred_list = predicted[i]
        pred_set = set()
        if len(pred_list) != tgt_len or len(act_set) != tgt_len:
            continue
        total_user += 1

        for j in range(tgt_len):  # tgt_len is number of history data
            pred_position = pred_list[j][:topk]
            rank = 100000000  # value 100000000 is INF
            truth = actual[i][j]
            for p in range(tgt_len):
                pred_position_ = pred_list[p][:topk]
                for k in range(len(pred_position_)):
                    if (pred_position_[k] == truth):
                        rank = min(rank, k)
                        break
            if rank != 100000000:
                MRR += 1.0 / (rank + 1.0)
                NDCG += 1.0 / np.log2(rank + 2.0)
            pred_set = pred_set | set(pred_position)

        Recall += len(act_set & pred_set) / float(len(act_set))
    MRR /= tgt_len
    NDCG /= tgt_len
    return NDCG / total_user, MRR / total_user, Recall / total_user
