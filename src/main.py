import torch
from omegaconf import OmegaConf
import yaml
from datetime import datetime
import warnings
import argparse
import json
import pdb
import os

from utils.utils import set_seed, log_setting, version_build
from data_provider.dataloader import get_dataloader
from model import build_model



warnings.filterwarnings('ignore')


def main(sweep_config=None):
    """
    main experiment

    Parameters
    ----------
    sweep_config : dict
    """
    # arg parser
    model_name = args.model
    dataset = args.dataname
    subdataname = args.subdataname

    # process init
    now = datetime.now()
    if subdataname is not None:
        group_name = f'{dataset}/{subdataname}-{model_name}'
        process_name = f'{group_name}-{now.strftime("%Y%m%d_%H%m%S")}'

    else:
        group_name = f'{dataset}-{model_name}'
        process_name = f'{group_name}-{now.strftime("%Y%m%d_%H%m%S")}'

    # savedir
    if subdataname is not None:
        logdir = os.path.join(config['log_dir'], f'{dataset}/{model_name}/{subdataname}')
    else:
        logdir = os.path.join(config['log_dir'], f'{dataset}/{model_name}')
    savedir = version_build(logdir=logdir, is_train=args.train, resume=args.resume)
    logger = log_setting(savedir, f'{now.strftime("%Y%m%d_%H%m%S")}')

    # save arguments
    json.dump(vars(args), open(os.path.join(savedir, 'arguments.json'), 'w'), indent=4)

    # multiple GPU init
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    # set seed
    set_seed(args.seed)

    model_params = config[model_name]
    scale = config['scale']

    # save model parameters
    json.dump(dict(model_params), open(os.path.join(savedir, 'model_params.json'), 'w'), indent=4)

    # model type init
    if model_name in ('VTTSAT', 'VTTPAT'):
        model_type = 'reconstruction'
    elif model_name in (''):
        model_type = 'prediction'
    else:
        model_type = 'classification'

    logger.info(f'Process {process_name} start!')

    data_info = OmegaConf.create(config[args.dataname])

    # load data
    trainloader, validloader, testloader, interloader = get_dataloader(data_name=args.dataname,
                                                                       sub_data_name=args.subdataname,
                                                                       data_info=data_info,
                                                                       loader_params=config['loader_params'],
                                                                       scale=scale,
                                                                       window_size=model_params.window_size,
                                                                       slide_size=int(model_params.window_size/2),
                                                                       model_type=model_type)


    # data shape
    b, model_params['window_size'], model_params['feature_num'] = next(iter(trainloader))['given'].shape
    logger.info(f'Data shape is ({b}, {model_params.window_size}, {model_params.feature_num})')

    # model build
    model = build_model(args, model_params, savedir)
    logger.info('Model build success!!')

    # training
    if args.train:
        logger.info('Training Start!')
        history = model.train(trainloader, validloader, testloader)
        for i in range(len(history['train_loss'])):
            train_loss = history['train_loss'][i]
            valid_loss = history['validation_loss'][i]
            # precision = history['precision'][i]
            # recall = history['recall'][i]
            # roc_auc = history['roc_auc'][i]
            # f1 = history['f1'][i]
            logger.info(f"Epoch: {i + 1} Train Loss: {train_loss:.7f} Vali Loss: {valid_loss:.7f} ")
                        # f"precision: {precision:.4f} recall: {recall:.4f} f1: {f1:.4f} ROC_AUC: {roc_auc:.4f}")
        logger.info('Model training success!!')

    # test
    if args.test:
        history = model.test(validloader, testloader)
        # for i in range(len(history['precision'])):
        #     precision = history['precision'][i]
        #     recall = history['recall'][i]
        #     roc_auc = history['roc_auc'][i]
        #     f1 = history['f1'][i]
            # logger.info(f"Result -> precision: {precision:.4f} recall: {recall:.4f} f1: {f1:.4f} ROC_AUC: {roc_auc:.4f}")

        logger.info('Model test success!!')

    torch.cuda.empty_cache()


if __name__ == '__main__':
    # set arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--train', action='store_true', help='training model')
    parser.add_argument('--test', action='store_true', help='anomaly scoring')
    parser.add_argument('--resume', type=int, default=None, help='version number to re-train or test model')
    parser.add_argument('--model', type=str, required=True,
                        choices=['VTTSAT', 'VTTPAT'],
                        help="model(VTTSAT, VTTPAT)")

    # data
    parser.add_argument('--dataname', type=str, required=True, help='data name(SWaT, SMD, SMAP, MSL)',
                        choices=['SWaT', 'SMD', 'SMAP', 'MSL'])
    parser.add_argument('--subdataname', type=str, default=None, help='dataset name')
    parser.add_argument('--window_size', type=int, default=100, help='window size for data loader')
    parser.add_argument('--slide_size', type=int, default=50, help='overlap ratio for data loader')

    # train options
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs')
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')

    # loss
    parser.add_argument('--loss', type=str, default='mse', choices=['mse', 'mae'],
                        help='select loss function')

    # setting
    parser.add_argument('--seed', type=int, default=72, help="set randomness")
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu number')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1', help='device ids of multiple gpus')
    parser.add_argument('--configure', type=str, default='config.yaml', help='configure file load')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for dataloader')

    args = parser.parse_args()

    # yaml file load
    with open(args.configure) as f:
        config = OmegaConf.load(f)
    
    # main
    main()

