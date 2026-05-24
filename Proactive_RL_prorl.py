import argparse
import os
from logging import getLogger

import torch
import numpy as np
import yaml
import time
from accelerate import Accelerator

from collator import Collator_RL
from model import PRARec
from trainer_RL_prorl import PRATrainer_ProRL  

from utils import *
from data_utils import *
import warnings

warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--config_file', type=str, required=True, help='Config file')
    parser.add_argument('--pretrained_ckpt', type=str, required=True, help='Path to pretrained checkpoint')
    parser.add_argument('--mode', type=str, required=True, choices=['prorl', 'eval'],
                        help='Training or evaluation mode')
    
    # ProRL hyperparameters
    parser.add_argument('--prorl_beta', type=float, default=None, help='KL weight')
    parser.add_argument('--prorl_num_samples', type=int, default=None, help='Sample size per rollout')
    parser.add_argument('--prorl_lr', type=float, default=None, help='RL learning rate')
    parser.add_argument('--prorl_epochs', type=int, default=None, help='RL training epochs')
    parser.add_argument('--prorl_gamma', type=float, default=None, help='Discount Factor')
    
    # Reward weight
    parser.add_argument('--reward_weight_ctr', type=float, default=None, help='CTR reward weight')
    parser.add_argument('--reward_weight_ioi', type=float, default=None, help='IOI reward weight')
    parser.add_argument('--reward_weight_ior', type=float, default=None, help='IOR reward weight')
    
    return parser.parse_known_args()



def main(config):

    print(f"Seed: {config['rand_seed']}")

    init_seed(config['rand_seed'], config['reproducibility'])
    init_logger(config)

    logger = getLogger()
    accelerator = config['accelerator']
    log(f'Device: {config["device"]}', accelerator, logger)
    log(f'Config: {str(config)}', accelerator, logger)
    log(f'ProRL Training Mode: Beta={config.get("prorl_beta", 0.01)}, Gamma={config.get("prorl_gamma", 0.99)}, Num Samples={config.get("prorl_num_samples", 4)}',
        accelerator, logger)

    tokenizers = get_tokenizers(config)

    train_dataset, valid_dataset, test_dataset = get_proactive_datasets(config)

    train_collate_fn = Collator_RL(config, tokenizers, 'train')
    test_collate_fn = Collator_RL(config, tokenizers, 'test')

    with accelerator.main_process_first():
        model = PRARec(config, train_dataset, tokenizers[-1])

    log(model, accelerator, logger)
    log(model.n_parameters, accelerator, logger)

    if config['pretrained_ckpt']:
        log(f'Loading pretrained model from {config["pretrained_ckpt"]}', accelerator, logger)
        checkpoint = torch.load(config['pretrained_ckpt'], map_location='cpu', weights_only=False)
        model_states = checkpoint['model'] if 'model' in checkpoint else checkpoint

        model_state_dict = model.state_dict()
        filtered_states = {}
        for k, v in model_states.items():
            if k in model_state_dict and model_state_dict[k].shape == v.shape:
                filtered_states[k] = v
            else:
                log(f'Skipping key {k} due to shape mismatch', accelerator, logger)

        model.load_state_dict(filtered_states, strict=False)
        log(f'Successfully loaded pretrained weights', accelerator, logger)

    train_data = get_dataloader(config, train_dataset, train_collate_fn, 'train')
    valid_data = get_dataloader(config, valid_dataset, test_collate_fn, 'valid')
    test_data = get_dataloader(config, test_dataset, test_collate_fn, 'test')

    trainer = PRATrainer_ProRL(config, model, tokenizers[-1], train_data)

    if config['mode'] == 'eval':
        log('Running evaluation only mode...', accelerator, logger)

        prorl_ckpt_path = config.get('prorl_checkpoint')
        if prorl_ckpt_path and os.path.exists(prorl_ckpt_path):
            trainer.load_states(prorl_ckpt_path)
            log(f'Loaded ProRL checkpoint from {prorl_ckpt_path}', accelerator, logger)

        # evaluate
        trainer.model, test_data = accelerator.prepare(trainer.model, test_data)
        test_results, all_results = trainer.evaluate_all_tokenizer(test_data, store=True)

        if accelerator.is_main_process:
            for key in test_results:
                accelerator.log({f'Test_Metric/{key}': test_results[key]})

        log(f'Test Results: {test_results}', accelerator, logger)
        for i, results in enumerate(all_results):
            log(f'Test Results {i}: {results}', accelerator, logger)

    else:
        # Training
        log('Starting ProRL training...', accelerator, logger)

        prorl_epochs = config.get('prorl_epochs', 10)

        early_stopping = trainer.fit(train_data, test_data, prorl_epochs, epoch_bias=0)

    trainer.end()


if __name__ == '__main__':

    args, unparsed_args = parse_args()
    command_line_configs = parse_command_line_args(unparsed_args)

    # Config
    config = {}
    config.update(yaml.safe_load(open(args.config_file, 'r')))
    config.update(command_line_configs)

    config['pretrained_ckpt'] = args.pretrained_ckpt
    config['mode'] = args.mode
    config['dataset'] = args.dataset

    # Hyperparameters
    if args.prorl_beta is not None:
        config['prorl_beta'] = args.prorl_beta
    if args.prorl_num_samples is not None:
        config['prorl_num_samples'] = args.prorl_num_samples
    if args.prorl_lr is not None:
        config['prorl_lr'] = args.prorl_lr
    if args.prorl_epochs is not None:
        config['prorl_epochs'] = args.prorl_epochs
    if args.prorl_gamma is not None:
        config['prorl_gamma'] = args.prorl_gamma
    
    # Reward weight for ctr, ioi, ior
    if args.reward_weight_ctr is not None:
        config['reward_weight_ctr'] = args.reward_weight_ctr
    if args.reward_weight_ioi is not None:
        config['reward_weight_ioi'] = args.reward_weight_ioi
    if args.reward_weight_ior is not None:
        config['reward_weight_ior'] = args.reward_weight_ior

    config['run_local_time'] = get_local_time()

    ckpt_name = get_file_name(config) + f'_prorl_lr_{config.get("prorl_lr", 1e-5)}_prorl_beta_{config.get("prorl_beta", 0.01)}_prorl_gamma_{config.get("prorl_gamma", 0.99)}'
    config['ckpt_name'] = ckpt_name
    config['data_dir'] = os.path.join(config['data_dir'], config['dataset'])
    config['ckpt_dir'] = os.path.join(config['ckpt_dir'], config['dataset'], ckpt_name)
    config['log_dir'] = os.path.join(config['log_dir'], config['dataset'], ckpt_name) 
    config['tensorboard_log_dir'] = os.path.join(config['tensorboard_log_dir'], config['dataset'], ckpt_name) 

    config = convert_config_dict(config)

    config['device'], config['use_ddp'] = init_device()
    config['accelerator'] = Accelerator()

    print("config info: ", config)

    if config['use_ddp']:
        torch.distributed.barrier(device_ids=[int(os.environ['LOCAL_RANK'])])

    main(config)