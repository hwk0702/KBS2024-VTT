from utils.utils import set_seed, log_setting, version_build
from data_provider.dataloader import get_dataloader
from model import build_model
import torch
from omegaconf import OmegaConf
import yaml
import warnings
import argparse
import json
import pdb
import os

warnings.filterwarnings('ignore')


def main():
    # arg parser
    model_name = args.model
    dataset = args.dataname
    subdataname = args.subdataname

    if subdataname is not None:
        logdir = os.path.join(config['log_dir'], f'{dataset}/{model_name}/{subdataname}')
    else:
        logdir = os.path.join(config['log_dir'], f'{dataset}/{model_name}')

    if subdataname is not None:
        logdir = os.path.join(config['log_dir'], f'{dataset}/{model_name}/{subdataname}')
    else:
        logdir = os.path.join(config['log_dir'], f'{dataset}/{model_name}')
    savedir = version_build(logdir=logdir, is_train=False, resume=args.resume)
    logger = log_setting(savedir, f'XAI')
    # model_params = json.loads(args.params)
    # with open(os.path.join(savedir, 'model_params.json'), 'r') as j:
    #     model_params = json.load(j)
    model_params = OmegaConf.load(os.path.join(savedir, 'model_params.json'))
    # args = json.loads(args.arg)
    # with open(os.path.join(savedir, 'arguments.json'), 'r') as j:
    #     args2 = json.load(j)
    args2 = OmegaConf.load(os.path.join(savedir, 'arguments.json'))
    args2.resume = args.resume

    # model type init
    if model_name in ('VTTSAT', 'VTTPAT'):
        model_type = 'reconstruction'
    elif model_name in (''):
        model_type = 'prediction'
    else:
        model_type = 'classification'


    data_info = OmegaConf.create(config[args.dataname])

    # pdb.set_trace()

    # load data
    _, _, testloader = get_dataloader(data_name=args.dataname,
                                      sub_data_name=args.subdataname,
                                      data_info=data_info,
                                      loader_params=config['loader_params'],
                                      scale=model_params['scale'],
                                      window_size=args2.window_size,
                                      slide_size=args2.slide_size,
                                      model_type=model_type)

    # data shape
    b, w, model_params['feature_num'] = next(iter(testloader))['given'].shape
    logger.info(f'Data shape is ({b}, {w}, {model_params.feature_num})')

    # model build
    model = build_model(args2, model_params, savedir)
    logger.info('Model build success!!')

    logger.info('interpret1 Start!')
    model.interpret1(testloader)
    logger.info('interpret1 Finish!')
    logger.info('interpret2 Start!')
    model.interpret2(testloader)
    logger.info('interpret2 Finish!')


if __name__ == '__main__':
    # set arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--resume', type=int, required=True, default=None,
                        help='version number to re-train or test model')
    parser.add_argument('--model', type=str, required=True,
                        choices=['VTTSAT', 'VTTPAT'],
                        help="model(VTTSAT, VTTPAT)")

    # data
    parser.add_argument('--dataname', type=str, required=True, help='data name(SWaT, SMD, SMAP, MSL)',
                        choices=['SWaT', 'SMD', 'SMAP', 'MSL'])
    parser.add_argument('--subdataname', type=str, default=None, help='dataset name')

    # # load model parameters, arguments
    # parser.add_argument('--arg', type=str, required=True, help='argument json file')
    # parser.add_argument('--params', type=str, required=True, help='model parameters json file')

    args = parser.parse_args()
    config = OmegaConf.load('config.yaml')

    # main
    main()



