import os
import gc
import time
import random
import torch
import logging
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

# DDP related libraries
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from models import CMCM
from trains import CMCMTrainer
from data.load_data import MMDataLoader
from config.config_regression import ConfigRegression
from config.config_classification import ConfigClassification
from utils.functions import Storage

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def setup_ddp(rank, world_size):
    """初始化DDP进程组"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # 初始化进程组
    dist.init_process_group(
        backend='nccl',  # 使用NCCL后端（GPU通信）
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    torch.cuda.set_device(rank)

def cleanup_ddp():
    """清理DDP进程组"""
    dist.destroy_process_group()

def run(args, rank=0, world_size=1):
    logger = logging.getLogger()
    # 标记是否主进程，供下游 trainer / 其它模块使用，只允许主进程写文件和日志
    args.is_main_process = (rank == 0)

    # DDP模式下只在主进程创建目录
    if rank == 0:
        if not os.path.exists(args.model_save_dir):
            os.makedirs(args.model_save_dir)
    
    args.model_save_path = os.path.join(args.model_save_dir, f'{args.modelName}-{args.datasetName}-{args.train_mode}.pth')
    
    # 设置设备
    if args.use_ddp:
        device = torch.device(f'cuda:{rank}')
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    args.device = device
    args.rank = rank
    args.world_size = world_size
    
    if rank == 0:
        logger.info(f"Using device: {device}, DDP: {args.use_ddp}, World size: {world_size}")

    # 加载数据
    dataloader = MMDataLoader(args)

    # 自动推断分类任务的类别数
    if args.train_mode == 'classification' and getattr(args, 'output_head', 'mlp') == 'mlp' and (not hasattr(args, 'num_classes') or args.num_classes in (None, 0, -1)):
        dataset2num = {
            'meld': 7,
            'cherma': 7,
        }
        inferred = dataset2num.get(str(args.datasetName).lower())
        args.num_classes = int(inferred)
        if rank == 0:
            logger.info(f'Auto-inferred num_classes={args.num_classes} for dataset {args.datasetName}')

    # 创建模型
    model = CMCM(args).to(device)
    
    # 用DDP包装模型
    if args.use_ddp:
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    def print_trainable_parameters(model):
        # DDP模式下需要访问module
        _model = model.module if isinstance(model, DDP) else model
        trainable_params = 0
        all_param = 0
        for _, param in _model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        if rank == 0:
            logger.info(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}%")

    print_trainable_parameters(model)

    # 创建训练器
    trainer = CMCMTrainer(args)
    trainer.do_train(model, dataloader)
    
    # 单卡模式才一起测试
    if rank == 0 and not args.use_ddp:
        assert os.path.exists(args.model_save_path)
        # 重新创建模型用于测试
        test_model = CMCM(args).to(device)
        
        if os.path.exists(args.model_save_path):
            saved = torch.load(args.model_save_path, map_location=args.device)
            test_model.load_state_dict(saved, strict=False)
        else:
            logger.warning(f"Checkpoint {args.model_save_path} not found, skip reloading.")

        # 测试
        if args.tune_mode:
            results = trainer.do_test(test_model, dataloader['valid'], mode="VALID")
        else:
            results = trainer.do_test(test_model, dataloader['test'], mode="TEST")

        del test_model
        torch.cuda.empty_cache()
        gc.collect()
        
        return results
    else:
        return None

def test_only(args, rank=0, world_size=1):
    logger = set_log(args) # set log after rebuilding args
    logger = logging.getLogger()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args.device = device
    args.rank = rank
    args.world_size = world_size

    # dataloader
    dataloader = MMDataLoader(args)

    # 重新创建模型用于测试
    test_model = CMCM(args).to(device)

    if os.path.exists(args.model_save_path):
        saved = torch.load(args.model_save_path, map_location=args.device)
        test_model.load_state_dict(saved, strict=False)
    else:
        logger.warning(f"Checkpoint {args.model_save_path} not found, skip reloading.")
        return None

    trainer = CMCMTrainer(args)
    # 测试
    if args.tune_mode:
        results = trainer.do_test(test_model, dataloader['valid'], mode="VALID")
    else:
        results = trainer.do_test(test_model, dataloader['test'], mode="TEST")

    del test_model
    torch.cuda.empty_cache()
    gc.collect()
    return results


def run_ddp_worker(rank, world_size, args_cp):
    setup_ddp(rank, world_size)
    args_cp = Storage(args_cp)
    # 重建 args
    args_cp.device = torch.device(f'cuda:{rank}')
    args_cp.rank = rank
    args_cp.world_size = world_size
    args_cp.use_ddp = True
    logger = set_log(args_cp) # set log after rebuilding args
    try:
        result = run(args_cp, rank, world_size)
    finally:
        cleanup_ddp()
    return result

def run_normal(args):
    logger = set_log(args) # set log after rebuilding args
    args.res_save_dir = os.path.join(args.res_save_dir)
    init_args = args
    model_results = []
    seeds = args.seeds
    args.model_save_path = os.path.join(args.model_save_dir, f'{args.modelName}-{args.datasetName}-{args.train_mode}.pth')
    
    for i, seed in enumerate(seeds):
        args = init_args
        # 先根据当前 seed 和任务模式加载配置，得到完整的 args（含 dataPath 等）
        if args.train_mode == "regression":
            config = ConfigRegression(args)
        else:
            config = ConfigClassification(args)
        args = config.get_config()

        setup_seed(seed)
        args.seed = seed
        
        logger.info('Start running %s...' % (args.modelName))
        logger.info(args)
        args.cur_time = i + 1
        
        # 根据是否使用DDP选择不同的运行方式
        if args.use_ddp:
            import torch.multiprocessing as mp
            world_size = torch.cuda.device_count()
            logger.info(f"Starting DDP training with {world_size} GPUs")
            def to_spawn_cfg(args):
                cfg = dict(args)
                return cfg
            args_cp = to_spawn_cfg(args)
            mp.spawn(
                run_ddp_worker,
                args=(world_size, args_cp),
                nprocs=world_size,
                join=True
            )

            # DDP 训练已完成，这里只在主进程做一次加载+测试，不再重新训练
            test_results = test_only(args, rank=0, world_size=1)
        else:
            # 单卡：直接使用已经通过 Config* 完整配置后的 args
            test_results = run(args, rank=0, world_size=1)
        
        if test_results is not None:
            model_results.append(test_results)

    # 保存结果（只在主进程）
    if model_results:
        criterions = list(model_results[0].keys())
        save_path = os.path.join(args.res_save_dir, f'{args.datasetName}-{args.train_mode}-{args.warm_up_epochs}.csv')
        if not os.path.exists(args.res_save_dir):
            os.makedirs(args.res_save_dir)
        if os.path.exists(save_path):
            df = pd.read_csv(save_path)
        else:
            df = pd.DataFrame(columns=["Model", "Seed"] + criterions)
        
        for i in range(len(model_results)):
            res = [args.modelName, f'{seeds[i]}']
            for c in criterions:
                res.append(round(model_results[i][c] * 100, 2))
            df.loc[len(df)] = res
        
        df.to_csv(save_path, index=None)
        logger.info('Results are added to %s...' % (save_path))

def set_log(args):
    log_file_path = f'logs/{args.modelName}-{args.datasetName}.log'
    logger = logging.getLogger() 
    logger.setLevel(logging.INFO)

    for ph in logger.handlers:
        logger.removeHandler(ph)
    formatter_file = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
    # 确保日志目录存在
    os.makedirs('logs', exist_ok=True)
    
    fh = logging.FileHandler(log_file_path)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter_file)
    logger.addHandler(fh)
    formatter_stream = logging.Formatter('%(message)s')
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter_stream)
    logger.addHandler(ch)
    return logger

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_mode', type=str, default="regression",
                        help='regression / classification')
    parser.add_argument('--modelName', type=str, default='cmcm',
                        help='support CMCM')
    parser.add_argument('--datasetName', type=str, default='simsv2',
                        help='support mosi/mosei/simsv2/iemocap/meld/cherma')
    parser.add_argument('--root_dataset_dir', type=str, default='/home/chenxiyao/code/MSE-Adapter/Multimodal-dataset/',
                        help='Location of the root directory where the dataset is stored')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='num workers of loading data')
    parser.add_argument('--model_save_dir', type=str, default='results/models',
                        help='path to save results.')
    parser.add_argument('--res_save_dir', type=str, default='results/results',
                        help='path to save results.')
    parser.add_argument('--pretrain_LM', type=str, default='/home/chenxiyao/code/MSE-Adapter/MLLMs/Qwen-1.8B/',
                        help='path to load pretrain LLM.')
    parser.add_argument('--fusion_strategy', type=str, default='text_guided',
                        help='text_guided | tri_fusion')
    parser.add_argument('--output_head', type=str, default='llm',
                        help='llm | mlp')
    parser.add_argument('--num_classes', type=int, default=7,
                        help='classification head num classes (used when output_head=mlp)')
    
    # DDP相关参数
    parser.add_argument('--use_ddp', action='store_true',
                        help='whether to use DDP for multi-GPU training')
    
    return parser.parse_args()

if __name__ == '__main__':
    """
    # 单卡训练（原有方式）
    python run.py

    # 多卡DDP训练（自动使用所有可见GPU）
    python run.py --use_ddp

    # 指定使用特定GPU
    CUDA_VISIBLE_DEVICES=0,1,2,3 python run.py --use_ddp
    """
    args = parse_args()
    logger = set_log(args)
    
    # 如果使用DDP，设置多进程启动方法
    if args.use_ddp:
        import torch.multiprocessing as mp
        mp.set_start_method('spawn', force=True)

    for data_name in ['simsv2', 'mosei', 'meld', 'cherma']:
    # for data_name in ['simsv2']:
        if data_name in ['mosi', 'mosei', 'sims', 'simsv2']:
            args.train_mode = 'regression'
        else:
            args.train_mode = 'classification'

        args.datasetName = data_name
        args.seeds = [1111, 2222, 3333, 4444, 5555]
        # args.seeds = [1111]
        run_normal(args)