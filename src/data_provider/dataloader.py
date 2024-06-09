import torch
from data_provider.dataset import BuildDataset, load_dataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def get_dataloader(
        data_name: str,
        sub_data_name: str,
        data_info: dict,
        loader_params: dict,
        scale: str = None,
        window_size: int = 60,
        slide_size: int = 30,
        model_type: str = 'reconstruction',
) -> [torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Return data loader

    Parameters
    ----------
    data_name : str
        name of the data
    sub_data_name : str
        name of the sub data (using only for SMD, SMAP and MSL data)
    data_info : dict
        information of data from config.yaml
    loader_params : dict
        parameters of data loader
    scale : str(default= None)
        scaling method name ex) minmax
    window_size : int(default=60)
        window size for time series condition
    slide_size : int(default=30)
        moving window size
    model_type : str(default= reconstruction)
        model type (reconstruction, prediction)

    Returns
    -------
    train/dev/tst dataloader : torch.utils.data.DataLoader
        given : torch.Tensor (shape=(batch size, window size, num of features))
            input time-series data
        ts : ndarray (shape=(batch size, window size))
            input timestamp
        answer : torch.Tensor (shape=(batch size, window size, num of features))
            target time-series data
            If model_type is prediction, window size is 1
        attack : ndarray
            attack labels

    """
    assert scale in (None, 'minmax', 'minmax_square', 'minmax_m1p1', 'standard')

    # load dataset (data, timestamp, label)
    trn, trn_ts, dev, dev_ts, tst, tst_ts, label = load_dataset(dataname=data_name,
                                                                datainfo=data_info,
                                                                subdataname=sub_data_name)

    # scaling (minmax, minmax square, minmax m1p1, standard)
    if scale is not None:
        if scale == 'minmax':
            scaler = MinMaxScaler(feature_range=(-1, 1))
            scaler.fit(trn)
            trn = scaler.transform(trn)
            dev = scaler.transform(dev)
            tst = scaler.transform(tst)
        elif scale == 'minmax_square':
            scaler = MinMaxScaler()
            scaler.fit(trn)
            trn = scaler.transform(trn) ** 2
            dev = scaler.transform(dev) ** 2
            tst = scaler.transform(tst) ** 2
        elif scale == 'minmax_m1p1':
            trn = 2 * (trn / trn.max(axis=0)) - 1
            dev = 2 * (dev / dev.max(axis=0)) - 1
            tst = 2 * (tst / tst.max(axis=0)) - 1
        elif scale == 'standard':
            scaler = StandardScaler()
            scaler.fit(trn)
            trn = scaler.transform(trn)
            dev = scaler.transform(dev)
            tst = scaler.transform(tst)
        print(f'{scale} Normalization done')

    # build dataset
    trn_dataset = BuildDataset(trn, trn_ts, window_size, slide_size,
                               attacks=None, model_type=model_type)

    dev_dataset = BuildDataset(dev, dev_ts, window_size, slide_size,
                               attacks=None, model_type=model_type)

    tst_dataset = BuildDataset(tst, tst_ts, window_size, window_size,
                               attacks=label, model_type=model_type)

    # torch dataloader
    trn_dataloader = torch.utils.data.DataLoader(trn_dataset,
                                                 batch_size=loader_params['batch_size'],
                                                 shuffle=loader_params['shuffle'],
                                                 num_workers=loader_params['num_workers'],
                                                 pin_memory=loader_params['pin_memory'],
                                                 drop_last=False)

    dev_dataloader = torch.utils.data.DataLoader(dev_dataset,
                                                 batch_size=loader_params['batch_size'],
                                                 shuffle=loader_params['shuffle'],
                                                 num_workers=loader_params['num_workers'],
                                                 pin_memory=loader_params['pin_memory'],
                                                 drop_last=False)

    tst_dataloader = torch.utils.data.DataLoader(tst_dataset,
                                                 batch_size=loader_params['batch_size'],
                                                 shuffle=loader_params['shuffle'],
                                                 num_workers=loader_params['num_workers'],
                                                 pin_memory=loader_params['pin_memory'],
                                                 drop_last=False)

    inter_dataloader = torch.utils.data.DataLoader(tst_dataset,
                                                 batch_size=1,
                                                 shuffle=loader_params['shuffle'],
                                                 num_workers=loader_params['num_workers'],
                                                 pin_memory=loader_params['pin_memory'],
                                                 drop_last=False)

    return trn_dataloader, dev_dataloader, tst_dataloader, inter_dataloader
