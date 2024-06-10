from models import VTTPAT, VTTSAT
from utils.tools import EarlyStopping, adjust_learning_rate, check_graph
from utils.utils import load_model, CheckPoint
from utils.metrics import bf_search

import torch
import torch.nn as nn
from torch import optim
from einops import rearrange

import os
import time

import matplotlib.pyplot as plt
import warnings
import numpy as np
import json
import pdb

warnings.filterwarnings('ignore')

class build_model():
    """
    Build and train or test a model

    Parameters
    ----------
    args : dict
        arguments
    params : dict
        model's hyper parameters
    savedir : str
        save directory

    Attributes
    ------------
    args : dict
        arguments
    params : dict
        model's hyper parameters
    savedir : str
        save directory
    device : torch.device
        device
    model : nn.module
        model

    Methods
    -------
    _build_model()
        Select the model you want to train or test and build it
        + multi gpu setting
    _acquire_device()
        check gpu usage
    _select_optimizer()
        select the optimizer (default AdamW)
    _select_criterion()
        select the criterion (default MSELoss)
    valid()

    train()

    test()


    """

    def __init__(self, args, params, savedir):
        super(build_model, self).__init__()
        self.args = args
        self.params = params
        self.savedir = savedir
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        model_dict = {
            'VTTSAT': VTTSAT,
            'VTTPAT': VTTPAT,
        }
        
        model = model_dict[self.args.model].Model(self.params).float()
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            print('using multi-gpu')
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _select_optimizer(self):
        if self.params.optim == 'adamw':
            model_optim = optim.AdamW(self.model.parameters(), lr=self.params.lr)
        elif self.params.optim == 'adam':
            model_optim = optim.Adam(self.model.parameters(), lr=self.params.lr)
        elif self.params.optim == 'sgd':
            model_optim = optim.SGD(self.model.parameters(), lr=self.params.lr)
        return model_optim

    def _select_criterion(self):
        if self.args.loss == 'mse':
            criterion = nn.MSELoss(reduction='none')
        elif self.args.loss == 'mae':
            criterion = nn.L1Loss(reduction='none')
        return criterion
    
    def valid(self, valid_loader, criterion, epoch):
        """
        validation

        Parameters
        ----------
        valid_loader : torch.utils.data.DataLoader
            given : torch.Tensor (shape=(batch size, window size, num of features))
                input time-series data
            ts : ndarray (shape=(batch size, window size))
                input timestamp
            answer : torch.Tensor (shape=(batch size, window size, num of features))
                target time-series data
                If model_type is prediction, window size is 1

        criterion

        Return
        ------
        total_loss

        """
        total_loss = []
        valid_score = []
        self.model.eval()
        with torch.no_grad():               
            for batch_idx, batch in enumerate(valid_loader):
                batch_x = batch['given'].float().to(self.device)
                batch_y = batch['answer'].float().to(self.device)

                if self.args.model in ['VTTSAT', 'VTTPAT']:
                    output, _ = self.model(batch_x)
                    loss = criterion(output, batch_y)
                    valid_score.append(np.mean(loss.cpu().detach().numpy(), axis=2))
                    loss = torch.mean(loss)
                else:
                    output = self.model(batch_x)
                    loss = criterion(output, batch_y)
                    valid_score.append(np.mean(loss.cpu().detach().numpy(), axis=2))
                    loss = torch.mean(loss)
                
                total_loss.append(loss.item())

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss, valid_score

    def train(self, train_loader, valid_loader, test_loader, alpha=.5, beta=.5):
        """
        training

        Parameters
        ----------
        train_loader : torch.utils.data.DataLoader
            given : torch.Tensor (shape=(batch size, window size, num of features))
                input time-series data
            ts : ndarray (shape=(batch size, window size))
                input timestamp
            answer : torch.Tensor (shape=(batch size, window size, num of features))
                target time-series data
                If model_type is prediction, window size is 1
        valid_loader
            given : torch.Tensor (shape=(batch size, window size, num of features))
                input time-series data
            ts : ndarray (shape=(batch size, window size))
                input timestamp
            answer : torch.Tensor (shape=(batch size, window size, num of features))
                target time-series data
                If model_type is prediction, window size is 1
        test_loader
            given : torch.Tensor (shape=(batch size, window size, num of features))
                input time-series data
            ts : ndarray (shape=(batch size, window size))
                input timestamp
            answer : torch.Tensor (shape=(batch size, window size, num of features))
                target time-series data
                If model_type is prediction, window size is 1
            attack : ndarray
            attack labels

        Return
        ------
        model

        """
        time_now = time.time()
        best_metrics = None
        
        if self.args.resume is not None:
            print(f'resume version{self.args.resume}')
            weights, start_epoch, self.args.lr, best_metrics = load_model(resume=self.args.resume,
                                                                          logdir=self.savedir)
            if isinstance(self.model, nn.DataParallel):
                self.model.module.load_state_dict(weights)
            else:
                self.model.load_state_dict(weights, map_location=self.device)

        # set checkpoint
        ckp = CheckPoint(logdir=self.savedir,
                         last_metrics=best_metrics,
                         metric_type='loss')

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        history = {'train_loss': [], 'validation_loss': []}

        for epoch in range(self.args.epochs):
            iter_count = 0
            train_loss = []
            train_score = []

            self.model.train()
            epoch_time = time.time()
            
            for batch_idx, batch in enumerate(train_loader):
                iter_count += 1
                batch_x = batch['given'].float().to(self.device)
                batch_y = batch['answer'].float().to(self.device)

                model_optim.zero_grad()
                if self.args.model in ['VTTSAT', 'VTTPAT']:
                    output, _ = self.model(batch_x)
                else:
                    output = self.model(batch_x)
                    
                loss = criterion(output, batch_y)
                train_score.append(loss.cpu().detach().numpy().mean(axis=2))
                loss = torch.mean(loss)
                train_loss.append(loss.item())
                loss.backward()
                model_optim.step()

                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((self.args.epochs - epoch) * train_steps - batch_idx)

            train_score = np.concatenate(train_score).flatten()
            train_loss_avg = np.average(train_loss)
            valid_loss, valid_score = self.valid(valid_loader, criterion, epoch)
            valid_score = np.concatenate(valid_score).flatten()
            
            dist, attack = self.inference(test_loader, epoch)

            # result save
            folder_path = os.path.join(self.savedir, 'results', f'epoch_{epoch}')
            os.makedirs(folder_path, exist_ok=True)
            visual = check_graph(dist, attack, piece=4)
            visual.savefig(os.path.join(folder_path, f'graph.png'))
            np.save(os.path.join(folder_path, f'dist.npy'), dist)
            np.save(os.path.join(folder_path, f'attack.npy'), attack)

            print(f"Epoch: {epoch + 1}, Steps: {train_steps} cost time: {time.time() - epoch_time} | "
                  f"Train Loss: {train_loss_avg:.7f} Vali Loss: {valid_loss:.7f} ")

            history['train_loss'].append(train_loss_avg)
            history['validation_loss'].append(valid_loss)

            ckp.check(epoch=epoch + 1, model=self.model, score=valid_loss, lr=model_optim.param_groups[0]['lr'])
            adjust_learning_rate(model_optim, epoch + 1, self.params)

            if early_stopping.validate(valid_loss):
                print("Early stopping")
                break

        best_model_path = os.path.join(self.savedir, f'{ckp.best_epoch}.pth')

        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(torch.load(best_model_path)['weight'])
        else:
            self.model.load_state_dict(torch.load(best_model_path)['weight'])

        return history


    def test(self, valid_loader, test_loader, alpha=.5, beta=.5):
        """
        test

        Parameters
        ----------
        test_loader
            given : torch.Tensor (shape=(batch size, window size, num of features))
                input time-series data
            ts : ndarray (shape=(batch size, window size))
                input timestamp
            answer : torch.Tensor (shape=(batch size, window size, num of features))
                target time-series data
                If model_type is prediction, window size is 1
            attack : ndarray
            attack labels

        Returns
        -------

        """

        if self.args.resume is not None and self.args.train is not True:
            print(f'resume version{self.args.resume}')
            weights, start_epoch, self.args.lr, best_metrics = load_model(resume=self.args.resume,
                                                                          logdir=self.savedir)
            if isinstance(self.model, nn.DataParallel):
                self.model.module.load_state_dict(weights)
            else:
                self.model.load_state_dict(weights)

        dist, attack, pred = [], [], []
        history = dict()
        criterion = self._select_criterion()

        self.model.eval()
        with torch.no_grad():      
            for batch_idx, batch in enumerate(test_loader):
                batch_x = batch['given'].float().to(self.device)
                batch_y = batch['answer'].float().to(self.device)
                batch_attack = batch['attack'].reshape(-1, batch['attack'].shape[-1]).numpy()

                if self.args.model in ['VTTSAT', 'VTTPAT']:
                    predictions, _ = self.model(batch_x)
                else:
                    predictions = self.model(batch_x)

                score = criterion(predictions, batch_y).cpu().detach().numpy()
                pred.append(predictions.cpu().detach().numpy())
                dist.append(np.mean(score, axis=2))
                attack.append(batch_attack)

        dist = np.concatenate(dist).flatten()
        attack = np.concatenate(attack).flatten()
        pred = np.concatenate(pred)

        # score
        scores = dist.copy()
        K = [0, 1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        f1_values = []

        print(f'Threshold start: {np.percentile(scores, 90):.4f} end: {np.percentile(scores, 99):.4f}')

        for k in K:
            scores = dist.copy()
            [f1, precision, recall, _, _, _, _, roc_auc, _, _], threshold = bf_search(scores, attack,
                                                                                start=np.percentile(scores, 50),
                                                                                end=np.percentile(scores, 99),
                                                                                step_num=1000,
                                                                                K=k,
                                                                                verbose=False)

            f1_values.append(f1)
            print(f"K: {k} precision: {precision:.4f} recall: {recall:.4f} f1: {f1:.4f} AUROC: {roc_auc:.4f}")
            history.setdefault(f'precision_{k}', []).append(precision)
            history.setdefault(f'recall_{k}', []).append(recall)
            history.setdefault(f'f1_{k}', []).append(f1)
            history.setdefault(f'roc_auc', []).append(roc_auc)

        auc = sum(0.5 * (f1_values[i] + f1_values[i + 1]) * (K[i + 1] - K[i]) for i in range(len(K) - 1)) / 100
        print(f'PA%K AUC: {auc}')

        visual = check_graph(dist, attack, piece=4, threshold=threshold)
        figure_path = os.path.join(self.savedir, 'fig')
        os.makedirs(figure_path, exist_ok=True)
        visual.savefig(os.path.join(figure_path, f'whole.png'))

        # result save
        folder_path = os.path.join(self.savedir, 'results')
        os.makedirs(folder_path, exist_ok=True)

        np.save(os.path.join(folder_path, f'dist.npy'), dist)
        np.save(os.path.join(folder_path, f'attack.npy'), attack)
        np.save(os.path.join(folder_path, f'pred.npy'), pred)

        return history

    def inference(self, test_loader, epoch, alpha=.5, beta=.5):
        """
        inference

        Parameters
        ----------
        test_loader
            given : torch.Tensor (shape=(batch size, window size, num of features))
                input time-series data
            ts : ndarray (shape=(batch size, window size))
                input timestamp
            answer : torch.Tensor (shape=(batch size, window size, num of features))
                target time-series data
                If model_type is prediction, window size is 1
            attack : ndarray
            attack labels
        epoch : int
            train epoch

        Returns
        -------

        """
        dist = []
        attack = []
        criterion = self._select_criterion()

        self.model.eval()

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                batch_x = batch['given'].float().to(self.device)
                batch_y = batch['answer'].float().to(self.device)

                if self.args.model in ['VTTSAT', 'VTTPAT']:
                    predictions, _ = self.model.forward(batch_x)
                    score = criterion(predictions, batch_y).cpu().detach().numpy()
                    dist.append(np.mean(score, axis=2))
                    
                else:
                    predictions = self.model.forward(batch_x)
                    score = criterion(predictions, batch_y).cpu().detach().numpy()
                    dist.append(np.mean(score, axis=2))

                attack.append(batch['attack'].reshape(-1, batch['attack'].shape[-1]).numpy())

        dist = np.concatenate(dist).flatten()
        attack = np.concatenate(attack).flatten()

        return dist, attack


    def interpret(self, test_loader):
        """
        interpret

        Parameters
        ----------
        test_loader
            given : torch.Tensor (shape=(batch size, window size, num of features))
                input time-series data
            ts : ndarray (shape=(batch size, window size))
                input timestamp
            answer : torch.Tensor (shape=(batch size, window size, num of features))
                target time-series data
                If model_type is prediction, window size is 1
            attack : ndarray
            attack labels

        Returns
        -------

        """
        if self.args.resume is not None:
            print(f'resume version{self.args.resume}')
            weights, start_epoch, self.args.lr, best_metrics = load_model(resume=self.args.resume,
                                                                          logdir=self.savedir)
            if isinstance(self.model, nn.DataParallel):
                self.model.module.load_state_dict(weights)
            else:
                self.model.load_state_dict(weights)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        hist = dict()
        
        scores, labels, tss, vattns, tattns = [], [], [], [], []
        
        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                batch_x = batch['given'].float().to(self.device)
                batch_y = batch['answer'].float().to(self.device)

                predictions, [prior_vattn, prior_tattn] = self.model(batch_x, use_attn=True)
                _, [post_vattn, post_tattn] = self.model(predictions, use_attn=True)
                
                score = criterion(predictions, batch_y).cpu().detach().numpy()
                scores.append(score)
                labels.append(label)
                tss.append(batch_idx)

                vattns.append([
                    [vt.cpu().detach().numpy() for vt in prior_vattn],
                    [vt.cpu().detach().numpy() for vt in post_vattn]
                ])
                tattns.append([
                    [tt.cpu().detach().numpy() for tt in prior_tattn],
                    [tt.cpu().detach().numpy() for tt in post_tattn]
                ])

        return scores, labels, tss, vattns, tattns