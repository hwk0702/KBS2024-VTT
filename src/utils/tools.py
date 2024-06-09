import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy
import scipy.cluster.vq
import scipy.spatial.distance
dst = scipy.spatial.distance.euclidean

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, params):
    """
    adjust learning rate

    Parameters
    ----------
    optimizer
    epoch
    params
    """
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if params.lradj == 'type1':
        lr_adjust = {epoch: params.lr * (0.5 ** ((epoch - 1) // 1))}
    elif params.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    """
    early stopping

    Parameters
    ----------
    patience : int
    verbose : bool

    Methods
    -------
    validate(loss)
    """
    def __init__(self, patience: int = 0, verbose: bool = 0):
        self._step = 0
        self._loss = float('inf')
        self.patience = patience
        self.verbose = verbose

    def validate(self, loss):
        if self._loss < loss:
            self._step += 1
            if self._step > self.patience:
                if self.verbose:
                    print(f'\n Training process is stopped early....')
                return True
        else:
            self._step = 0
            self._loss = loss

        return False

#
# def check_graph(xs, att, piece=1, threshold=None):
#     """
#     anomaly score and anomaly label visualization
#
#     Parameters
#     ----------
#     xs : np.ndarray
#         anomaly scores
#     att : np.ndarray
#         anomaly labels
#     piece : int
#         number of figures to separate
#     threshold : float(default=None)
#         anomaly threshold
#
#     Return
#     ------
#     fig : plt.figure
#     """
#     l = xs.shape[0]
#     chunk = l // piece
#     fig, axs = plt.subplots(piece, figsize=(20, 4 * piece))
#     for i in range(piece):
#         L = i * chunk
#         R = min(L + chunk, l)
#         xticks = range(L, R)
#         axs[i].plot(xticks, xs[L:R])
#         if len(xs[L:R]) > 0:
#             peak = max(xs[L:R])
#             axs[i].plot(xticks, att[L:R] * peak * 0.3)
#         if threshold is not None:
#             axs[i].axhline(y=threshold, color='r')
#
#     return fig


def check_graph(xs, att, piece=1, threshold=None):
    """
    anomaly score and anomaly label visualization

    Parameters
    ----------
    xs : np.ndarray
        anomaly scores
    att : np.ndarray
        anomaly labels
    piece : int
        number of figures to separate
    threshold : float(default=None)
        anomaly threshold

    Return
    ------
    fig : plt.figure
    """
    l = xs.shape[0]
    chunk = l // piece
    fig, axs = plt.subplots(piece, figsize=(20, 4 * piece))
    for i in range(piece):
        L = i * chunk
        R = min(L + chunk, l)
        xticks = np.arange(L, R)
        if piece == 1:
            ax = axs
        else:
            ax = axs[i]
        ax.plot(xticks, xs[L:R], color='#0C090A')
        ymin, ymax = ax.get_ylim()
        ymin = 0
        ax.set_ylim(ymin, ymax)
        if len(xs[L:R]) > 0:
            ax.vlines(xticks[np.where(att[L:R] == 1)], ymin=ymin, ymax=ymax, color='#FED8B1',
                      alpha=0.6, label='true anomaly')
        ax.plot(xticks, xs[L:R], color='#0C090A', label='anomaly score')
        if threshold is not None:
            ax.axhline(y=threshold, color='r', linestyle='--', alpha=0.8, label=f'threshold:{threshold:.4f}')
        ax.legend()

    return fig


def gap(data, refs=None, nrefs=20, ks=range(1, 11)):
    """
    Compute the Gap statistic for an nxm dataset in data.
    Either give a precomputed set of reference distributions in refs as an (n,m,k) scipy array,
    or state the number k of reference distributions in nrefs for automatic generation with a
    uniformed distribution within the bounding box of data.
    Give the list of k-values for which you want to compute the statistic in ks.
    """
    shape = data.shape
    if refs is None:
        tops = data.max(axis=0)
        bots = data.min(axis=0)
        dists = scipy.matrix(scipy.diag(tops - bots))

        rands = scipy.random.random_sample(size=(shape[0], shape[1], nrefs))
        for i in range(nrefs):
            rands[:, :, i] = rands[:, :, i] * dists + bots
    else:
        rands = refs

    gaps = scipy.zeros((len(ks),))
    for (i, k) in enumerate(ks):
        (kmc, kml) = scipy.cluster.vq.kmeans2(data, k)
        disp = sum([dst(data[m, :], kmc[kml[m], :]) for m in range(shape[0])])

        refdisps = scipy.zeros((rands.shape[2],))
        for j in range(rands.shape[2]):
            (kmc, kml) = scipy.cluster.vq.kmeans2(rands[:, :, j], k)
            refdisps[j] = sum([dst(rands[m, :, j], kmc[kml[m], :]) for m in range(shape[0])])
        gaps[i] = scipy.log(scipy.mean(refdisps)) - scipy.log(disp)
    return gaps