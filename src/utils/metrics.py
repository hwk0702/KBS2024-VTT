import numpy as np
from sklearn.metrics import f1_score, roc_curve, roc_auc_score, precision_score, recall_score, average_precision_score
import sklearn.metrics
from collections import Counter

def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe


def calc_point2point(predict, actual):
    """
    calculate f1 score by predict and actual.
    Args:
        predict (np.ndarray): the predict label
        actual (np.ndarray): np.ndarray
    """
    TP = np.sum(predict * actual)
    TN = np.sum((1 - predict) * (1 - actual))
    FP = np.sum(predict * (1 - actual))
    FN = np.sum((1 - predict) * actual)
    precision = TP / (TP + FP + 0.00001)
    recall = TP / (TP + FN + 0.00001)
    f1 = 2 * precision * recall / (precision + recall + 0.00001)
    return f1, precision, recall, TP, TN, FP, FN


def adjust_predicts(score, label,
                    threshold=None,
                    pred=None,
                    calc_latency=False):
    """
    Calculate adjusted predict labels using given `score`, `threshold` (or given `pred`) and `label`.
    Args:
        score (np.ndarray): The anomaly score
        label (np.ndarray): The ground-truth label
        threshold (float): The threshold of anomaly score.
            A point is labeled as "anomaly" if its score is lower than the threshold.
        pred (np.ndarray or None): if not None, adjust `pred` and ignore `score` and `threshold`,
        calc_latency (bool):
    Returns:
        np.ndarray: predict labels
    """
    if len(score) != len(label):
        raise ValueError("score and label must have the same length")
    score = np.asarray(score)
    label = np.asarray(label)
    latency = 0
    if pred is None:
        predict = score > threshold
    else:
        predict = pred
    actual = label > 0.1
    anomaly_state = False
    anomaly_count = 0
    for i in range(len(score)):
        if actual[i] and predict[i] and not anomaly_state:
                anomaly_state = True
                anomaly_count += 1
                for j in range(i, 0, -1):
                    if not actual[j]:
                        break
                    else:
                        if not predict[j]:
                            predict[j] = True
                            latency += 1
        elif not actual[i]:
            anomaly_state = False
        if anomaly_state:
            predict[i] = True
    if calc_latency:
        return predict, latency / (anomaly_count + 1e-4)
    else:
        return predict


def PA_percentile(score, label,
                  threshold=None,
                  pred=None,
                  K=100,
                  calc_latency=False):
    """
    Calculate adjusted predict labels using given `score`, `threshold` (or given `pred`) and `label`.
    Args:
        score (np.ndarray): The anomaly score
        label (np.ndarray): The ground-truth label
        threshold (float): The threshold of anomaly score.
            A point is labeled as "anomaly" if its score is lower than the threshold.
        pred (np.ndarray or None): if not None, adjust `pred` and ignore `score` and `threshold`,
        calc_latency (bool):
    Returns:
        np.ndarray: predict labels
    """
    if len(score) != len(label):
        raise ValueError("score and label must have the same length")
    score = np.asarray(score)
    label = np.asarray(label)
    latency = 0
    if pred is None:
        predict = score > threshold
    else:
        predict = pred
    actual = label > 0.1
    anomaly_state = False
    anomaly_count = 0
    anomalies = []

    for i in range(len(actual)):
        if actual[i]:
            if not anomaly_state:
                anomaly_state = True
                anomaly_count += 1
                anomalies.append([i, i])
            else:
                anomalies[-1][-1] = i
        else:
            anomaly_state = False

    for i, [start, end] in enumerate(anomalies):
        collect = Counter(predict[start:end + 1])[1]
        anomaly_count += collect
        collect_ratio = collect / (end - start + 1)

        if collect_ratio * 100 >= K and collect > 0:
            predict[start:end + 1] = True
            latency += (end - start + 1) - collect

    if calc_latency:
        return predict, latency / (anomaly_count + 1e-4)
    else:
        return predict


def calc_seq(score, label, threshold, K=0, calc_latency=False):
    """
    Calculate f1 score for a score sequence
    """
    if calc_latency:
        roc_auc = roc_auc_score(label, score)
        auprc = average_precision_score(label, score)
        #predict, latency = adjust_predicts(score, label, threshold, calc_latency=calc_latency)
        predict, latency = PA_percentile(score, label, threshold, K=K, calc_latency=calc_latency)
        t = list(calc_point2point(predict, label))
        t.append(roc_auc)
        t.append(auprc)
        t.append(latency)
        return t
    else:
        roc_auc = roc_auc_score(label, score)
        auprc = average_precision_score(label, score)
        # predict = adjust_predicts(score, label, threshold, calc_latency=calc_latency)
        predict = PA_percentile(score, label, threshold, K=K, calc_latency=calc_latency)
        t = list(calc_point2point(predict, label))
        t.append(roc_auc)
        t.append(auprc)
        return t


def bf_search(score, label, start, end=None, step_num=1, display_freq=1, K=0, verbose=True) -> object:
    """
    Find the best-f1 score by searching best `threshold` in [`start`, `end`).
    Returns:
        list: list for results
        float: the `threshold` for best-f1
    """
    if step_num is None or end is None:
        end = start
        step_num = 1
    search_step, search_range, search_lower_bound = step_num, end - start, start
    if verbose:
        print("search range: ", search_lower_bound, search_lower_bound + search_range)
    threshold = search_lower_bound
    m = (-1., -1., -1.)
    m_t = 0.0
    for i in range(search_step):
        threshold += search_range / float(search_step)
        target = calc_seq(score, label, threshold, K=K, calc_latency=True)
        if target[0] > m[0]:
            m_t = threshold
            m = target
        if verbose and i % display_freq == 0:
            print("cur thr: ", threshold, target, m, m_t)
    return m, m_t

def valid_search(valid_score, score, label, start, end=None, interval=0.1, display_freq=1, K=0, verbose=True) -> object:
    """
    Find the best-f1 score by searching best `threshold` in [`start`, `end`).
    Returns:
        list: list for results
        float: the `threshold` for best-f1
    """
    if step_num is None or end is None:
        end = start
        interval = 1
    search_interval, search_range, search_lower_bound = interval, end - start, start
    if verbose:
        print("search range: ", search_lower_bound, search_lower_bound + search_range)
    threshold = search_lower_bound
    m = (-1., -1., -1.)
    m_t = 0.0
    for i in range(search_range // search_interval):
        threshold = np.percentile(valid_score, 100-(i+1)*search_interval)
        target = calc_seq(score, label, threshold, K=K, calc_latency=True)
        if target[0] > m[0]:
            m_t = threshold
            m = target
        if verbose and i % display_freq == 0:
            print("cur thr: ", threshold, target, m, m_t)
    return m, m_t

def pot_eval(init_score, score, label, q=1e-3, level=0.02):
    """
    Run POT method on given score.
    Args:
        init_score (np.ndarray): The data to get init threshold.
            For `OmniAnomaly`, it should be the anomaly score of train set.
        score (np.ndarray): The data to run POT method.
            For `OmniAnomaly`, it should be the anomaly score of test set.
        label:
        q (float): Detection level (risk)
        level (float): Probability associated with the initial threshold t
    Returns:
        dict: pot result dict
    """
    s = SPOT(q)  # SPOT object
    s.fit(init_score, score)  # data import
    s.initialize(level=level, min_extrema=False, verbose=False)  # initialization step
    ret = s.run(dynamic=False)  # run
    print(len(ret['alarms']))
    print(len(ret['thresholds']))
    pot_th = np.mean(ret['thresholds'])
    pred, p_latency = adjust_predicts(score, label, pot_th, calc_latency=True)
    p_t = calc_point2point(pred, label)
    print('POT result: ', p_t, pot_th, p_latency)
    return {
        'pot-f1': p_t[0],
        'pot-precision': p_t[1],
        'pot-recall': p_t[2],
        'pot-TP': p_t[3],
        'pot-TN': p_t[4],
        'pot-FP': p_t[5],
        'pot-FN': p_t[6],
        'pot-threshold': pot_th,
        'pot-latency': p_latency,
        'pot-ROC/AUC': p_t[7]
    }


# here for our refined best-f1 search method
def get_best_f1(score, label):
    """
    :param score: 1-D array, input score, tot_length
    :param label: 1-D array, standard label for anomaly
    :return: list for results, threshold
    """

    assert score.shape == label.shape
    print('***computing best f1***')
    search_set = []
    tot_anomaly = 0
    for i in range(label.shape[0]):
        tot_anomaly += (label[i] > 0.5)
    flag = 0
    cur_anomaly_len = 0
    cur_min_anomaly_score = 1e5
    for i in range(label.shape[0]):
        if label[i] > 0.5:
            # here for an anomaly
            if flag == 1:
                cur_anomaly_len += 1
                cur_min_anomaly_score = score[i] if score[i] < cur_min_anomaly_score else cur_min_anomaly_score
            else:
                flag = 1
                cur_anomaly_len = 1
                cur_min_anomaly_score = score[i]
        else:
            # here for normal points
            if flag == 1:
                flag = 0
                search_set.append((cur_min_anomaly_score, cur_anomaly_len, True))
                search_set.append((score[i], 1, False))
            else:
                search_set.append((score[i], 1, False))
    if flag == 1:
        search_set.append((cur_min_anomaly_score, cur_anomaly_len, True))
    search_set.sort(key=lambda x: x[0])
    best_f1_res = - 1
    threshold = 1
    P = 0
    TP = 0
    best_P = 0
    best_TP = 0
    for i in range(len(search_set)):
        P += search_set[i][1]
        if search_set[i][2]:  # for an anomaly point
            TP += search_set[i][1]
        precision = TP / (P + 1e-5)
        recall = TP / (tot_anomaly + 1e-5)
        f1 = 2 * precision * recall / (precision + recall + 1e-5)
        if f1 > best_f1_res:
            best_f1_res = f1
            threshold = search_set[i][0]
            best_P = P
            best_TP = TP

    print('***  best_f1  ***: ', best_f1_res)
    print('*** threshold ***: ', threshold)
    return (best_f1_res,
            best_TP / (best_P + 1e-5),
            best_TP / (tot_anomaly + 1e-5),
            best_TP,
            score.shape[0] - best_P - tot_anomaly + best_TP,
            best_P - best_TP,
            tot_anomaly - best_TP), threshold


# calculate evaluation metrics (best-F1, AUROC, AP) under point-adjust approach.
def get_adjusted_composite_metrics(score, label):
    score = -score  # change the recons prob to anomaly score, higher anomaly score means more anomalous
    # adjust the score for segment detection. i.e., for each ground-truth anomaly segment, use the maximum score
    # as the score of all points in that segment. This corresponds to point-adjust f1-score.
    assert len(score) == len(label)
    splits = np.where(label[1:] != label[:-1])[0] + 1
    is_anomaly = label[0] == 1
    pos = 0
    for sp in splits:
        if is_anomaly:
            score[pos:sp] = np.max(score[pos:sp])
        is_anomaly = not is_anomaly
        pos = sp
    sp = len(label)
    if is_anomaly:
        score[pos:sp] = np.max(score[pos:sp])

    # now get to adjust score for segment evaluation.
    fpr, tpr, threshold = sklearn.metrics.roc_curve(y_true=label, y_score=score, drop_intermediate=False)
    auroc = sklearn.metrics.auc(fpr, tpr)
    precision, recall, _ = sklearn.metrics.precision_recall_curve(y_true=label, probas_pred=score)
    # validate best f1
    f1 = np.max(2 * precision * recall / (precision + recall + 1e-5))
    ap = sklearn.metrics.average_precision_score(y_true=label, y_score=score, average=None)
    return auroc, ap, f1, precision, recall, fpr, tpr, threshold


def anomaly_metric(scores, true):
    fpr, tpr, thresholds = roc_curve(true, scores, pos_label=1)
    J = tpr - fpr
    ix = np.argmax(J)
    pred = np.where(scores < thresholds[ix], 0, 1)
    precision = precision_score(true, pred, pos_label=1)
    recall = recall_score(true, pred, pos_label=1)
    f1 = f1_score(true, pred, pos_label=1, average='micro')
    auroc = roc_auc_score(true, pred)

    return precision, recall, f1, auroc

def percentile_search(combined_energy, score, label, anomaly_ratio):
    threshold = np.percentile(combined_energy, 100 - anomaly_ratio)
    target = calc_seq(score, label, threshold, calc_latency=True)
    return target, threshold