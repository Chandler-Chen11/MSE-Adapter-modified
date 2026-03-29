import os
import time
import logging
import math
import copy
import argparse
import numpy as np
import pickle as plk
from glob import glob
from tqdm import tqdm
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch import optim
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from utils.functions import dict_to_str
from utils.metricsTop import MetricsTop
from transformers import get_cosine_schedule_with_warmup
import matplotlib.pyplot as plt
import matplotlib
from itertools import chain

logger = logging.getLogger()

class CMCMTrainer():
    def __init__(self, args):
        self.args = args
        self.args.tasks = "M"
        self.metrics = MetricsTop(args).getMetics(args.datasetName)

        # 特征映射（用于某些高级功能，DDP模式下需要注意同步）
        self.feature_map = {
            'fusion': torch.zeros(args.train_samples, args.post_fusion_dim, requires_grad=False).to(args.device),
            'text': torch.zeros(args.train_samples, args.post_text_dim, requires_grad=False).to(args.device),
            'audio': torch.zeros(args.train_samples, args.post_audio_dim, requires_grad=False).to(args.device),
            'vision': torch.zeros(args.train_samples, args.post_video_dim, requires_grad=False).to(args.device),
        }

        self.dim_map = {
            'fusion': torch.tensor(args.post_fusion_dim).float(),
            'text': torch.tensor(args.post_text_dim).float(),
            'audio': torch.tensor(args.post_audio_dim).float(),
            'vision': torch.tensor(args.post_video_dim).float(),
        }
        
        self.label_map = {
            'fusion': torch.zeros(args.train_samples, requires_grad=False).to(args.device),
            'text': torch.zeros(args.train_samples, requires_grad=False).to(args.device),
            'audio': torch.zeros(args.train_samples, requires_grad=False).to(args.device),
            'vision': torch.zeros(args.train_samples, requires_grad=False).to(args.device)
        }

        self.name_map = {
            'M': 'fusion',
            'T': 'text',
            'A': 'audio',
            'V': 'vision'
        }

    def do_train(self, model, dataloader):
        """训练函数，支持DDP"""
        rank = self.args.rank
        use_ddp = self.args.use_ddp
        
        scaler = GradScaler()
        
        # 获取需要优化的参数
        if isinstance(model, DDP):
            params = model.module.parameters()
        else:
            params = model.parameters()
            
        optimizer = optim.AdamW(params, lr=self.args.learning_rate, eps=1e-4)
        
        total_steps = len(dataloader['train']) * self.args.warm_up_epochs
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=0.1*total_steps, num_training_steps=total_steps)

        if rank == 0:
            logger.info("Start training...")

        epochs, best_epoch = 0, 0
        losses = []
        lr = []
        
        min_or_max = 'min' if self.args.KeyEval in ['MAE'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0
        
        # 获取训练采样器（DDP模式）
        train_sampler = dataloader.get('train_sampler', None)
        
        while True: 
            epochs += 1
            
            print(f"rank={rank}, epoch={epochs} train loop start, len(train_loader)={len(dataloader['train'])}")


            # DDP模式下设置epoch以确保每个epoch的数据shuffle不同
            if use_ddp and train_sampler is not None:
                train_sampler.set_epoch(epochs)
            
            # 训练阶段
            model.train()
            train_loss = 0.0
            left_epochs = self.args.update_epochs
            # 只在主进程显示进度条
            iterator = tqdm(dataloader['train'], disable=(rank != 0)) if rank == 0 else dataloader['train']
            for batch_data in iterator:
                if left_epochs == self.args.update_epochs:
                    optimizer.zero_grad()
                left_epochs -= 1

                vision = batch_data['vision'].to(self.args.device)
                audio = batch_data['audio'].to(self.args.device)
                text = batch_data['text'].to(self.args.device)
                
                if self.args.train_mode == 'regression':
                    labels_m = batch_data['labels']['M'].view(-1).to(self.args.device)
                else:
                    labels_m = batch_data['labels']['M'].to(self.args.device)

                if not self.args.need_data_aligned:
                    text_lengths = batch_data['text_lengths'].to(self.args.device)
                    audio_lengths = batch_data['audio_lengths'].to(self.args.device)
                    vision_lengths = batch_data['vision_lengths'].to(self.args.device)

                # 前向传播
                with autocast():
                    output = model(labels_m, (text, text_lengths), (audio, audio_lengths), (vision, vision_lengths))
                    loss = output['Loss']

                # 反向传播
                scaler.scale(loss).backward()
                train_loss += loss.item()
                lr.append(optimizer.state_dict()['param_groups'][0]['lr'])
                
                # 更新参数
                if not left_epochs:
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    left_epochs = self.args.update_epochs
                    
            # 最后一个batch的更新
            if not left_epochs:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                left_epochs = self.args.update_epochs
            
            # DDP模式下同步loss（可选）
            if use_ddp:
                train_loss_tensor = torch.tensor(train_loss).to(self.args.device)
                dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
                train_loss = train_loss_tensor.item() / self.args.world_size
            
            train_loss = train_loss / len(dataloader['train'])

            if rank == 0:
                logger.info("TRAIN-(%s) (%d/%d/%d)>> loss: %.4f" % (
                    self.args.modelName, epochs-best_epoch, epochs, self.args.cur_time, train_loss))
                losses.append(train_loss)

            # 验证阶段 & 早停逻辑
            stop_flag = False  # 由 rank0 决定，后面广播

            if epochs >= 1 and rank == 0:
                # 获取原始模型（非DDP包装）
                eval_model = model.module if isinstance(model, DDP) else model
                val_results = self.do_test(eval_model, dataloader['valid'], mode="VAL")
                cur_valid = val_results[self.args.KeyEval]
                
                isBetter = (
                    cur_valid <= (best_valid - 1e-6)
                    if min_or_max == 'min'
                    else cur_valid >= (best_valid + 1e-6)
                )
                if isBetter:
                    best_valid, best_epoch = cur_valid, epochs
                    self.save_model(model, epochs, self.args.model_save_path)

                # 只在 rank0 上根据 best_epoch 判断是否早停
                if epochs - best_epoch >= self.args.early_stop:
                    logger.info(f"Early stopping at epoch {epochs}")
                    stop_flag = True

            if use_ddp:
                # 将 rank0 的 stop_flag 广播给所有 rank
                stop_tensor = torch.tensor(
                    [int(stop_flag)], device=self.args.device
                )
                dist.broadcast(stop_tensor, src=0)
                if stop_tensor.item() == 1:
                    break
            else:
                # 单卡 / 非 DDP
                if stop_flag:
                    break
            

    def do_test(self, model, dataloader, mode="VAL"):
        """测试函数"""
        model.eval()
        y_pred = {'M': [], 'T': [], 'A': [], 'V': []}
        y_true = {'M': [], 'T': [], 'A': [], 'V': []}
        
        if self.args.train_mode == 'regression':
            with torch.no_grad():
                with tqdm(dataloader) as td:
                    for batch_data in td:
                        vision = batch_data['vision'].to(self.args.device)
                        audio = batch_data['audio'].to(self.args.device)
                        text = batch_data['text'].to(self.args.device)
                        
                        if not self.args.need_data_aligned:
                            text_lengths = batch_data['text_lengths'].to(self.args.device)
                            audio_lengths = batch_data['audio_lengths'].to(self.args.device)
                            vision_lengths = batch_data['vision_lengths'].to(self.args.device)
                            
                        with autocast():
                            outputs = model.generate((text, text_lengths), (audio, audio_lengths), (vision, vision_lengths))

                        if isinstance(outputs, list):
                            predict_label = torch.tensor(outputs).to(self.args.device)
                        else:
                            predict_label = outputs.to(self.args.device)

                        labels_m = batch_data['labels']['M'].view(-1).to(self.args.device)
                        y_pred['M'].append(predict_label.cpu())
                        y_true['M'].append(labels_m.cpu())
                        
            pred, true = torch.cat(y_pred['M']), torch.cat(y_true['M'])
            logger.info(mode + "-(%s)" % self.args.modelName + " >>")
            eval_results = self.metrics(pred, true)
            logger.info('M: >> ' + dict_to_str(eval_results))
        else:
            # 分类模式
            with torch.no_grad():
                with tqdm(dataloader) as td:
                    for batch_data in td:
                        vision = batch_data['vision'].to(self.args.device)
                        audio = batch_data['audio'].to(self.args.device)
                        text = batch_data['text'].to(self.args.device)
                        
                        if not self.args.need_data_aligned:
                            text_lengths = batch_data['text_lengths'].to(self.args.device)
                            audio_lengths = batch_data['audio_lengths'].to(self.args.device)
                            vision_lengths = batch_data['vision_lengths'].to(self.args.device)
                            
                        with autocast():
                            outputs = model.generate((text, text_lengths), (audio, audio_lengths), (vision, vision_lengths))

                        if isinstance(outputs, list):
                            predict_label = torch.tensor(outputs).to(self.args.device)
                        else:
                            predict_label = outputs

                        labels_m = batch_data['labels']['M']
                        y_pred['M'].append(predict_label)
                        y_true['M'].append(labels_m)
                        
            pred, true = list(chain(*y_pred['M'])), list(chain(*y_true['M']))
            eval_results = self.metrics(pred, true)
            logger.info(mode + "-(%s)" % self.args.modelName + " >>")
            logger.info('M: >> ' + dict_to_str(eval_results))

        return eval_results
    
    def l1_loss(self, y_pred, y_true, indexes=None, mode='fusion'):
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)
        if mode == 'fusion':
            loss = torch.mean(torch.abs(y_pred - y_true))
        return loss

    def init_labels(self, indexes, m_labels):
        self.label_map['fusion'][indexes] = m_labels
        self.label_map['text'][indexes] = m_labels
        self.label_map['audio'][indexes] = m_labels
        self.label_map['vision'][indexes] = m_labels

    def save_model(self, model, epoch, save_path):
        """保存模型，处理DDP包装"""
        # 获取原始模型（非DDP包装）
        if isinstance(model, DDP):
            model_to_save = model.module
        else:
            model_to_save = model
            
        param_grad_dic = {k: v.requires_grad for (k, v) in model_to_save.named_parameters()}
        
        # 只保存需要梯度的参数
        state_dict = {}
        for k, v in model_to_save.state_dict().items():
            if k in param_grad_dic and not param_grad_dic[k]:
                continue
            state_dict[k] = v.detach().cpu()
            
        logging.info(f"Saving checkpoint at epoch {epoch} to {save_path}.")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(state_dict, save_path)