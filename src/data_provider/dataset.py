from torch.utils.data import Dataset
import torch
import pandas as pd
import os
import numpy as np
from pathlib import Path
from datetime import timedelta
import dateutil

def dataframe_from_csv(target):
    return pd.read_csv(target).rename(columns=lambda x: x.strip())

def dataframe_from_csvs(targets):
    return pd.concat([dataframe_from_csv(x) for x in targets])

class BuildDataset(Dataset):
    """
    Build Dataset
    Parameters
    ----------
    data : ndarray(dtype=float, ndim=2, shape=(time, num of features))
        time-series data
    timestamps : ndarray
        time-series data's timestamp
    window_size : int
        window size for time series condition
    slide_size : int(default=1)
        moving window size
    attacks : ndarray(dtype=np.int, ndim=2, shape=(time,))
        attack label
    model_type : str(default=reconstruction)
        model type (reconstruction, prediction)
    Attributes
    ----------
    ts : ndarray
        time-series data's timestamp
    tag_values : ndarray(dtype=np.float32, ndim=2, shape=(time, num of features))
        time-series data
    window_size : int
        window size for time series condition
    model_type : str
        model type (reconstruction, prediction)
    valid_idxs : list
        first index of data divided by window
    Methods
    -------
    __len__()
        return num of valid windows
    __getitem__(idx)
        return data(given, ts, answer, attack)
    """

    def __init__(self,
                 data: np.ndarray,
                 timestamps: np.ndarray,
                 window_size: int,
                 slide_size: int = 1,
                 attacks: np.ndarray = None,
                 model_type: str = 'reconstruction'):
        self.ts = np.array(timestamps)
        self.tag_values = np.array(data, dtype=np.float32)
        self.window_size = window_size
        self.model_type = model_type

        self.valid_idxs = []
        if self.model_type == 'reconstruction':
            for L in range(0, len(self.ts) - window_size + 1, slide_size):
                R = L + window_size - 1
                try:
                    if dateutil.parser.parse(self.ts[R]) - dateutil.parser.parse(
                            self.ts[L]
                    ) == timedelta(seconds=window_size - 1):
                        self.valid_idxs.append(L)
                except:
                    if self.ts[R] - self.ts[L] == window_size - 1:
                        self.valid_idxs.append(L)
        elif self.model_type == 'prediction':
            for L in range(0, len(self.ts) - window_size, slide_size):
                R = L + window_size
                try:
                    if dateutil.parser.parse(self.ts[R]) - dateutil.parser.parse(
                            self.ts[L]
                    ) == timedelta(seconds=window_size):
                        self.valid_idxs.append(L)
                except:
                    if self.ts[R] - self.ts[L] == window_size:
                        self.valid_idxs.append(L)

        print(f"# of valid windows: {len(self.valid_idxs)}")

        if attacks is not None:
            self.attacks = np.array(attacks, dtype=np.float32)
            self.with_attack = True
        else:
            self.with_attack = False

    def __len__(self):
        return len(self.valid_idxs)

    def __getitem__(self, idx: str) -> dict:
        i = self.valid_idxs[idx]
        last = i + self.window_size
        item = {"given": torch.from_numpy(self.tag_values[i:last])}
        if self.model_type == 'reconstruction':
            # item["ts"] = self.ts[i:last]
            item["answer"] = torch.from_numpy(self.tag_values[i:last])
            if self.with_attack:
                item['attack'] = self.attacks[i:last]
        elif self.model_type == 'prediction':
            # item["ts"] = self.ts[last]
            item["answer"] = torch.from_numpy(self.tag_values[last])
            if self.with_attack:
                item['attack'] = self.attacks[last]
        return item


def load_dataset(dataname: str, datainfo: dict, subdataname: str = None, valid_split_rate: float = 0.8) -> \
        [np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    load dataset
    Parameters
    ----------
    dataname : str
        name of the data
    datainfo : dict
        information of data from config.yaml
    subdataname : str
        name of the sub data (using only for SMD, SMAP and MSL data)
    valid_split_rate : float(default=0.8)
        train, validation split rate
    Returns
    -------
    trainset : ndarray(shape=(time, num of features))
        train dataset
    train_timestamp : ndarray(shape=(time,))
        timestamp of train dataset
    validset : ndarray(shape=(time, num of features))
        validation dataset
    valid_timestamp : ndarray(shape=(time,))
        timestamp of validation dataset
    testset : ndarray(shape=(time, num of features))
        test dataset
    test_timestamp : ndarray(shape=(time,))
        timestamp of test dataset
    test_label : ndarray(shape=(time,))
        attack/anomaly labels of test dataset
    """
    try:
        assert dataname in ['SWaT', 'SMD', 'SMAP', 'MSL']
    except AssertionError as e:
        raise ValueError(f"Invalid dataname: {dataname}")

    if dataname == 'SWaT':
        trainset = pd.read_pickle(datainfo.train_path).drop(['Normal/Attack', ' Timestamp'], axis=1).to_numpy()
        valid_split_index = int(len(trainset) * valid_split_rate)
        validset = trainset[valid_split_index:]
        trainset = trainset[:valid_split_index]
        testset = pd.read_pickle(datainfo.test_path)
        train_timestamp = np.arange(len(trainset))
        valid_timestamp = np.arange(len(validset))
        test_timestamp = np.arange(len(testset))
        test_label = testset['Normal/Attack']
        test_label[test_label == 'Normal'] = 0
        test_label[test_label != 0] = 1
        testset = testset.drop(['Normal/Attack', ' Timestamp'],
                               axis=1).to_numpy()

    elif dataname == 'SMD':
        trainset = np.loadtxt(os.path.join(datainfo.train_dir, f'{subdataname}.txt'),
                              delimiter=',',
                              dtype=np.float32)
        valid_split_index = int(len(trainset) * valid_split_rate)
        validset = trainset[valid_split_index:]
        trainset = trainset[:valid_split_index]
        train_timestamp = np.arange(len(trainset))
        valid_timestamp = np.arange(len(validset))
        testset = np.loadtxt(os.path.join(datainfo.test_dir, f'{subdataname}.txt'),
                             delimiter=',',
                             dtype=np.float32)
        test_timestamp = np.arange(len(testset))
        test_label = np.loadtxt(os.path.join(datainfo.test_label_dir, f'{subdataname}.txt'),
                                delimiter=',',
                                dtype=np.int)

    else:
        trainset = np.load(os.path.join(datainfo.train_dir, f'{subdataname}.npy'))
        testset = np.load(os.path.join(datainfo.test_dir, f'{subdataname}.npy'))
        valid_split_index = int(len(trainset) * valid_split_rate)
        validset = trainset[valid_split_index:]
        trainset = trainset[:valid_split_index]
        train_timestamp = np.arange(len(trainset))
        valid_timestamp = np.arange(len(validset))
        test_timestamp = np.arange(len(testset))
        test_label_info = pd.read_csv(datainfo.test_label_path, index_col=0).loc[subdataname]
        test_label = np.zeros([int(test_label_info.num_values)], dtype=np.int)

        for i in eval(test_label_info.anomaly_sequences):
            if type(i) == list:
                test_label[i[0]:i[1] + 1] = 1
            else:
                test_label[i] = 1

    return trainset, train_timestamp, validset, valid_timestamp, testset, test_timestamp, test_label