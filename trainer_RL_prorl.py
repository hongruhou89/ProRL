import json
import os
import random
import math
from tqdm import tqdm
import numpy as np
import time
import random
import pickle
from collections import defaultdict, OrderedDict
from logging import getLogger
import copy
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from transformers.optimization import get_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from recbole.config import Config
from recbole.data.utils import create_dataset, data_preparation
from recbole.model.sequential_recommender import SASRec, BERT4Rec, FEARec, GRU4Rec, CORE, LightSANs
from transformers import get_linear_schedule_with_warmup

from model import AbstractModel
from tokenizer import Tokenizer
from evaluator import Evaluator, PRAEvaluator
from utils import *
from typing import Optional, Dict, Tuple, Union, List, Any

class MultiObjectiveRewardNormalizer:

    def __init__(self, config: dict):
        self.min_std = config.get('mo_min_std', 0.1)
        self.metrics = ['ioi', 'ior', 'ctr']
        
        self.mean = {m: 0.0 for m in self.metrics}
        self.std = {m: self.min_std for m in self.metrics}
        self.is_frozen = False
        
        self._count = {m: 0 for m in self.metrics}
        self._mean = {m: 0.0 for m in self.metrics}
        self._M2 = {m: 0.0 for m in self.metrics}

    def _welford_update(self, metric, new_values):
        """Welford batch update"""
        new_values = np.asarray(new_values)
        n_new = len(new_values)
        if n_new == 0:
            return
        
        new_mean = new_values.mean()
        new_M2 = ((new_values - new_mean) ** 2).sum()
        
        n_old = self._count[metric]
        n_total = n_old + n_new
        
        if n_old == 0:
            self._mean[metric] = float(new_mean)
            self._M2[metric] = float(new_M2)
        else:
            delta = new_mean - self._mean[metric]
            self._mean[metric] = (n_old * self._mean[metric] + n_new * new_mean) / n_total
            self._M2[metric] = self._M2[metric] + new_M2 + delta ** 2 * n_old * n_new / n_total
        
        self._count[metric] = n_total

    def collect_and_normalize(self, values_dict, valid_mask, stats_mask=None):
        if stats_mask is None:
            stats_mask = valid_mask
            
        normalized_dict = {}
        stats_dict = {}
        
        for metric in self.metrics:
            values = values_dict[metric]
            stats_values = values[stats_mask].detach().cpu().numpy()
            
            if len(stats_values) > 0:
                self._welford_update(metric, stats_values)
            
            mu = self._mean[metric]
            std = max(np.sqrt(self._M2[metric] / max(self._count[metric] - 1, 1)), self.min_std) if self._count[metric] > 1 else self.min_std

            # normalization
            normalized = (values - mu) / std
            
            normalized_dict[metric] = normalized
        
        return normalized_dict, stats_dict

    def freeze(self):
        for metric in self.metrics:
            self.mean[metric] = self._mean[metric]
            if self._count[metric] > 1:
                self.std[metric] = max(np.sqrt(self._M2[metric] / (self._count[metric] - 1)), self.min_std)
            else:
                self.std[metric] = self.min_std
        self.is_frozen = True

        self._count = None
        self._mean = None
        self._M2 = None

    def normalize(self, values_dict, valid_mask, stats_mask=None):
        normalized_dict = {}
        stats_dict = {}
        
        for metric in self.metrics:
            values = values_dict[metric]
            normalized = (values - self.mean[metric]) / self.std[metric]
            normalized_dict[metric] = normalized
        
        return normalized_dict, stats_dict

    def get_state(self):
        return {
            'mean': self.mean.copy(),
            'std': self.std.copy(),
            'is_frozen': self.is_frozen,
        }

    def load_state(self, state):
        self.mean = state['mean'].copy()
        self.std = state['std'].copy()
        self.is_frozen = state.get('is_frozen', True)

class PRATrainer_ProRL:

    def __init__(self, config: dict, model, tokenizer, train_dataloader: DataLoader):
        self.config = config
        self.accelerator = config['accelerator']
        self.logger = getLogger()

        # ==================== Model ====================
        self.model = model
        self.ref_model = copy.deepcopy(model)
        for param in self.ref_model.parameters():
            param.requires_grad = False
        self.tokenizer = tokenizer

        # ==================== Core hyperparameters ====================
        self.beta_kl = config.get('prorl_beta', 0.01)
        self.num_samples = config.get('prorl_num_samples', 16)
        self.max_proactive_length = config.get('max_proactive_length', 10)
        self.gamma = config.get('prorl_gamma', 1.0)  # discount factors

        # ==================== Multi-reward weights ====================
        self.reward_weight_ctr = config.get('reward_weight_ctr', 1)
        self.reward_weight_ioi = config.get('reward_weight_ioi', 1)
        self.reward_weight_ior = config.get('reward_weight_ior', 1)

        # Multi reward Calculator
        self.reward_normalizer = MultiObjectiveRewardNormalizer(config)

        # ==================== Early stopping ====================
        self.min_valid_ratio = config.get('min_valid_ratio', 0.3)
        self.valid_ratio_patience = config.get('valid_ratio_patience', 3)
        self.low_valid_count = 0

        # Structure parameters
        self.tokens_per_item = 4
        self.eos_token = tokenizer.eos_token

        # ==================== Evaluator ====================
        from evaluator import PRAEvaluator
        self.evaluator = PRAEvaluator(config, tokenizer)

        self.reward_computer = OptimizedRewardComputer_NoCOT(
            self.evaluator,
            self.tokenizer,
            {'device': self.accelerator.device, **self.config}
        )

        # ==================== Optimizer ====================
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.get('prorl_lr', 5e-6),
            weight_decay=config.get('weight_decay', 0.01)
        )

        train_dataloader = self.accelerator.prepare(train_dataloader)
        
        epochs = config.get('prorl_epochs', 10)
        steps_per_epoch = len(train_dataloader)
        total_steps = steps_per_epoch * epochs
        
        warmup_config = config.get('prorl_warmup_steps', 0.02)
        if isinstance(warmup_config, float) and warmup_config <= 1.0:
            num_warmup_steps = int(warmup_config * total_steps)
        else:
            num_warmup_steps = int(warmup_config)
        
        self.scheduler = get_scheduler(
            name="constant_with_warmup",
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_steps,
        )
        
        self.num_warmup_steps = num_warmup_steps
        self.total_steps = total_steps
        self.steps_per_epoch = steps_per_epoch

        self.model, self.ref_model, self.optimizer, self.scheduler = self.accelerator.prepare(
            self.model, self.ref_model, self.optimizer, self.scheduler
        )
        self.train_dataloader = train_dataloader
        self.ref_model.eval()

        # ==================== save path ====================
        self.saved_model_ckpt = os.path.join(
            config['ckpt_dir'],
            f'{config["ckpt_name"]}_prorl.pth'
        )
        self.results_dir = self.config['results_dir'] if self.config['results_dir'] else self.config['ckpt_dir']
        ensure_dir(self.results_dir)

        # ==================== training state ====================
        self.best_epoch = 0
        self.best_val_score = -1
        self.val_delay = config.get('val_delay', 0)
        self.writer = None

        os.makedirs(os.path.dirname(self.saved_model_ckpt), exist_ok=True)

    # ==================== Fused reward computation ====================
    def compute_fused_rewards(self, item_ioi, item_ior, item_ctr, num_items, duplicate_masks, valid_item_mask):
        """
        Args:
            item_ioi, item_ior, item_ctr: [batch_size, num_samples, max_items]
            num_items: [batch_size, num_samples]
            duplicate_masks: [batch_size, num_samples, max_items]
            valid_item_mask: [batch_size, num_samples, max_items]
        
        Returns:
            fused_rewards: [batch_size, num_samples, max_items]
            normalized_dict
            norm_stats
        """
        batch_size, num_samples, max_items = item_ioi.shape
        device = item_ioi.device

        # 1. hallucination & duplication analysis
        hallucination_mask = (item_ctr == 0) & valid_item_mask
        duplicate_mask = duplicate_masks & valid_item_mask & (~hallucination_mask)
        real_item_mask = valid_item_mask & (~hallucination_mask) & (~duplicate_mask)

        # 2. Effective items ratio
        valid_ioi = item_ioi[real_item_mask]
        valid_ior = item_ior[real_item_mask]
        if len(valid_ioi) > 0:
            min_ioi = valid_ioi.min().item()
            min_ior = valid_ior.min().item()
        else:
            min_ioi = 0.0
            min_ior = 0.0

        # 3. Reward process for hall & dup
        item_ctr_processed = item_ctr.clone()
        item_ioi_processed = item_ioi.clone()
        item_ior_processed = item_ior.clone()

        item_ctr_processed[hallucination_mask] = -1.0
        item_ioi_processed[hallucination_mask] = min_ioi - 1.0
        item_ior_processed[hallucination_mask] = min_ior - 1.0

        item_ctr_processed[duplicate_mask] = 0.0

        # 4. Multi-reward dict preparation
        raw_rewards = {
            'ioi': item_ioi_processed,
            'ior': item_ior_processed,
            'ctr': item_ctr_processed,
        }

        # 5. Nomalization
        if self.reward_normalizer.is_frozen:
            normalized_dict, norm_stats = self.reward_normalizer.normalize(
                raw_rewards, valid_item_mask, stats_mask=real_item_mask
            )
        else:
            normalized_dict, norm_stats = self.reward_normalizer.collect_and_normalize(
                raw_rewards, valid_item_mask, stats_mask=real_item_mask
            )

        fused_rewards = (
                self.reward_weight_ctr * normalized_dict['ctr'] / self.max_proactive_length +
                self.reward_weight_ioi * normalized_dict['ioi'] +
                self.reward_weight_ior * normalized_dict['ior']
        )

        # 6. avoid valid items
        fused_rewards = fused_rewards * valid_item_mask.float()

        return fused_rewards, normalized_dict, norm_stats, real_item_mask

    def parse_sequences_batch(self, sequences):
    
        batch_size, num_samples, seq_len = sequences.shape
        device = sequences.device

        flat_seqs = sequences.view(-1, seq_len)
        total_seqs = flat_seqs.shape[0]

        eos_mask = (flat_seqs == self.eos_token)
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(total_seqs, -1)

        # EOS position
        has_eos = eos_mask.any(dim=1)
        eos_pos = torch.where(has_eos,
                              (eos_mask * positions + (~eos_mask) * seq_len).min(dim=1)[0],
                              torch.full((total_seqs,), seq_len, device=device))

        # Items position range
        item_start = torch.zeros(total_seqs, dtype=torch.long, device=device)
        item_end = eos_pos
        num_tokens = torch.clamp(item_end - item_start, min=0)
        num_items = num_tokens // self.tokens_per_item

        # Is sequence valid (at least one item)
        valid = (num_items > 0).view(batch_size, num_samples)

        structures = {
            'valid': valid,
            'num_items': num_items.view(batch_size, num_samples),
            'item_start': item_start.view(batch_size, num_samples),
            'eos_pos': eos_pos.view(batch_size, num_samples),
            'has_eos': has_eos.view(batch_size, num_samples),
        }
        return structures

    def extract_items_batch(self, sequences, structures):
        batch_size, num_samples, seq_len = sequences.shape
        device = sequences.device
        tokens_per_item = self.tokens_per_item
        max_items = seq_len // tokens_per_item

        valid = structures['valid']
        num_items = structures['num_items']
        item_start = structures['item_start']

        item_offsets = torch.arange(max_items, device=device) * tokens_per_item
        token_offsets = torch.arange(tokens_per_item, device=device)

        positions = (
                item_start.unsqueeze(-1).unsqueeze(-1) +
                item_offsets.view(1, 1, -1, 1) +
                token_offsets.view(1, 1, 1, -1)
        )

        item_indices = torch.arange(max_items, device=device)
        tok_end_positions = item_start.unsqueeze(-1) + (item_indices + 1) * tokens_per_item
        in_range = tok_end_positions <= seq_len

        items_mask = (
                valid.unsqueeze(-1) &
                (item_indices.unsqueeze(0).unsqueeze(0) < num_items.unsqueeze(-1)) &
                in_range
        )

        positions_clamped = positions.clamp(0, seq_len - 1)
        flat_positions = positions_clamped.view(batch_size, num_samples, -1)
        items_flat = sequences.gather(dim=-1, index=flat_positions)
        items_tensor = items_flat.view(batch_size, num_samples, max_items, tokens_per_item)
        items_tensor = items_tensor * items_mask.unsqueeze(-1).long()

        return items_tensor, items_mask

    def detect_duplicates_batch(self, items_tensor, items_mask):
        batch_size, num_samples, max_items, tokens_per_item = items_tensor.shape
        device = items_tensor.device

        base = 65536
        multipliers = torch.tensor(
            [base ** i for i in range(tokens_per_item - 1, -1, -1)],
            device=device, dtype=torch.int64
        )

        item_hashes = (items_tensor.long() * multipliers).sum(dim=-1)
        item_hashes = torch.where(items_mask, item_hashes, torch.tensor(-1, dtype=torch.int64, device=device))

        hashes_row = item_hashes.unsqueeze(-1)
        hashes_col = item_hashes.unsqueeze(-2)

        indices = torch.arange(max_items, device=device)
        lower_tri = indices.unsqueeze(0) < indices.unsqueeze(1)

        is_equal = (hashes_row == hashes_col)
        is_duplicate = (is_equal & lower_tri.unsqueeze(0).unsqueeze(0)).any(dim=-1)

        duplicate_mask = is_duplicate & items_mask
        return duplicate_mask

    def zero_after_eos(self, tensor, target_value):
        seq_len = tensor.size(-1)
        positions = torch.arange(seq_len, device=tensor.device).expand_as(tensor)
        target_mask = (tensor == target_value)
        first_target_pos = torch.where(target_mask, positions, seq_len)
        first_target_pos = first_target_pos.min(dim=-1, keepdim=True)[0]
        zero_mask = positions > first_target_pos
        result = tensor.clone()
        result[zero_mask] = 0
        return result

    # ==================== Item reward computation ====================
    def compute_item_rewards(self, generated_sequences, batch):
        batch_size, num_samples, seq_len = generated_sequences.shape
        max_items = seq_len // self.tokens_per_item
        device = generated_sequences.device

        # 1. Sequence parse
        all_preds = self.zero_after_eos(generated_sequences, self.eos_token)
        structures = self.parse_sequences_batch(all_preds)
        items_tensor, items_mask = self.extract_items_batch(all_preds, structures)
        duplicate_masks = self.detect_duplicates_batch(items_tensor, items_mask)

        valid_mask = structures['valid']
        num_items = structures['num_items']

        # 2. Batch reward computation
        item_ioi, item_ior, item_ctr = self.reward_computer.compute_rewards_batch(
            items_tensor=items_tensor,
            items_mask=items_mask,
            valid_mask=valid_mask,
            num_items=num_items,
            proact_seman_ids=batch['proact_seman_id'],
            input_ids=batch['input_ids']
        )

        # 3. Item mask
        item_indices = torch.arange(max_items, device=device)
        valid_item_mask = (item_indices.unsqueeze(0).unsqueeze(0) < num_items.unsqueeze(-1)) & valid_mask.unsqueeze(-1)

        # 4. Fused reward computation
        fused_rewards, normalized_dict, norm_stats, real_item_mask = self.compute_fused_rewards(
            item_ioi, item_ior, item_ctr, num_items, duplicate_masks, valid_item_mask
        )

        # 5. Sequence Level Reward
        masked_ioi = item_ioi * valid_item_mask.float()
        masked_ior = item_ior * valid_item_mask.float()
        masked_ctr = item_ctr * valid_item_mask.float()
        masked_fused = fused_rewards * valid_item_mask.float()

        sequence_ioi = masked_ioi.sum(dim=-1)
        sequence_ior = masked_ior.sum(dim=-1)
        sequence_fused = masked_fused.sum(dim=-1)

        valid_counts = valid_item_mask.sum(dim=-1).clamp(min=1)
        sequence_ctr = masked_ctr.sum(dim=-1) / valid_counts

        valid_item_counts = ((item_ctr > 0) & valid_item_mask).sum(dim=-1).float()

        metrics = {
            'item_ioi': item_ioi,
            'item_ior': item_ior,
            'item_ctr': item_ctr,
            'fused_rewards': fused_rewards,
            'normalized_dict': normalized_dict,
            'norm_stats': norm_stats,
            'sequence_ioi': sequence_ioi,
            'sequence_ior': sequence_ior,
            'sequence_ctr': sequence_ctr,
            'sequence_fused': sequence_fused,
            'valid_mask': valid_mask,
            'duplicate_masks': duplicate_masks,
            'valid_item_counts': valid_item_counts,
            'items_mask': items_mask,
            'real_item_mask': real_item_mask,
        }

        return fused_rewards, structures, metrics

    # ==================== Advantage计算（新版本） ====================

    def compute_position_advantages(self, fused_rewards, structures, valid_mask):
        """
        Args:
            fused_rewards: [batch_size, num_samples, max_items]
            structures: Stucture infomation (dict)
            valid_mask: [batch_size, num_samples]
        
        Returns:
            position_advantages: [batch_size, num_samples, max_positions]
            position_active_mask: [batch_size, num_samples, max_positions]
            first_eos_mask: [batch_size, num_samples, max_positions]
        """
        batch_size, num_samples, max_items = fused_rewards.shape
        device = fused_rewards.device
        max_positions = self.max_proactive_length
        
        num_items = structures['num_items']  # [batch_size, num_samples]
        has_eos = structures['has_eos']  # [batch_size, num_samples]
        
        # ==================== 1. Position reward ====================
        position_rewards = torch.zeros(batch_size, num_samples, max_positions, device=device)
        pos_indices = torch.arange(max_positions, device=device).view(1, 1, -1)
        num_items_expanded = num_items.unsqueeze(-1)
        
        # Item Position：num_items > t
        is_item_position = pos_indices < num_items_expanded
        
        copy_len = min(max_items, max_positions)
        position_rewards[:, :, :copy_len] = fused_rewards[:, :, :copy_len]
        position_rewards = position_rewards * is_item_position.float()
        
        # Mask first EOS
        first_eos_mask = (pos_indices == num_items_expanded) & has_eos.unsqueeze(-1) & (num_items_expanded < max_positions)
        
        # ==================== 2. Return calculation ====================
        if self.gamma == 0:
            position_returns = position_rewards.clone()
        else:
            gamma_powers = self.gamma ** torch.arange(max_positions, device=device, dtype=torch.float32)
            gamma_powers = gamma_powers.view(1, 1, -1)
            
            weighted_rewards = position_rewards * gamma_powers
            reverse_cumsum = weighted_rewards.flip(-1).cumsum(-1).flip(-1)
            position_returns = reverse_cumsum / gamma_powers
        
        # ==================== 3. active mask ====================
        is_eos_position = (pos_indices == num_items_expanded) & has_eos.unsqueeze(-1)
        position_active_mask = (is_item_position | is_eos_position) & valid_mask.unsqueeze(-1)
        
        # ==================== 4. Baseline ====================
        # b_t = mean(G_t) over rollouts that reach position t
        active_counts = position_active_mask.float().sum(dim=1, keepdim=True).clamp(min=1)
        masked_returns = position_returns * position_active_mask.float()
        baseline = masked_returns.sum(dim=1, keepdim=True) / active_counts
        
        # ==================== 5. Advantage ====================
        position_advantages = (position_returns - baseline) * position_active_mask.float()
        
        return position_advantages, position_active_mask, first_eos_mask

    # ==================== Log Prob caculation====================

    def compute_position_log_probs(self, token_log_probs, structures, seq_len, batch_size, num_samples):
        """
        
        Args:
            token_log_probs: [batch_size * num_samples, seq_len] prob of each token
            structures: structure info
            seq_len: len of sequence generated
            batch_size
            num_samples: rollout size
        
        Returns:
            item_log_probs: [batch_size, num_samples, max_items] Item-level log probs
            first_eos_log_probs: [batch_size, num_samples] EOS prob
            item_valid_mask: [batch_size, num_samples, max_items] item mask
        """
        max_items = seq_len // self.tokens_per_item
        device = token_log_probs.device
        tokens_per_item = self.tokens_per_item

        token_log_probs_reshaped = token_log_probs.view(batch_size, num_samples, seq_len)

        valid = structures['valid']
        num_items = structures['num_items']
        item_start = structures['item_start']
        eos_pos = structures['eos_pos']
        has_eos = structures['has_eos']

        # =============== First EOS log_prob ===============
        eos_pos_clamped = eos_pos.clamp(0, seq_len - 1)
        first_eos_log_probs = token_log_probs_reshaped.gather(
            dim=-1,
            index=eos_pos_clamped.unsqueeze(-1)
        ).squeeze(-1)
        
        eos_valid = valid & has_eos & (eos_pos < seq_len)
        first_eos_log_probs = first_eos_log_probs * eos_valid.float()

        # =============== Item log_probs ===============
        item_offsets = torch.arange(max_items, device=device) * tokens_per_item
        token_offsets = torch.arange(tokens_per_item, device=device)

        positions = (
                item_start.unsqueeze(-1).unsqueeze(-1) +
                item_offsets.view(1, 1, -1, 1) +
                token_offsets.view(1, 1, 1, -1)
        )

        tok_end_positions = item_start.unsqueeze(-1) + (torch.arange(max_items, device=device) + 1) * tokens_per_item
        item_in_range = tok_end_positions <= seq_len

        positions_clamped = positions.clamp(0, seq_len - 1)
        flat_positions = positions_clamped.view(batch_size, num_samples, -1)
        gathered = token_log_probs_reshaped.gather(dim=-1, index=flat_positions)
        gathered = gathered.view(batch_size, num_samples, max_items, tokens_per_item)
        item_log_probs = gathered.mean(dim=-1)

        item_indices = torch.arange(max_items, device=device)
        item_valid_mask = (
                (item_indices.unsqueeze(0).unsqueeze(0) < num_items.unsqueeze(-1)) &
                valid.unsqueeze(-1) &
                item_in_range
        )
        item_log_probs = item_log_probs * item_valid_mask.float()

        return item_log_probs, first_eos_log_probs, item_valid_mask

    # ==================== Loss function ====================

    def compute_policy_loss(self, batch):
        batch_size = batch['input_ids'].shape[0]
        device = batch['input_ids'].device

        # 1. Rollouts
        with torch.no_grad():
            if self.config.get('use_ddp', False):
                model_t5 = self.model.module.t5
            else:
                model_t5 = self.model.t5
            model_t5.eval()

            # max_length = max_proactive_length * 4 (len semantic ids per item) + 1 (eos)
            max_gen_length = self.max_proactive_length * self.tokens_per_item + 1

            out = model_t5.generate(
                batch['input_ids'],
                num_beams=1,
                num_return_sequences=self.num_samples,
                do_sample=True,
                temperature=1.0,
                early_stopping=True,
                eos_token_id=self.eos_token,
                pad_token_id=0,
                max_length=max_gen_length,
                return_dict_in_generate=True,
                output_scores=True
            )

        generated_seqs = out.sequences[:, 1:].view(batch_size, self.num_samples, -1)
        seq_len = generated_seqs.shape[-1]

        # 2. Reward computation
        fused_rewards, structures, metrics = self.compute_item_rewards(generated_seqs, batch)

        # 3. Position-wise advantages estimation
        position_advantages, position_is_eos, first_eos_mask = self.compute_position_advantages(
            fused_rewards,
            structures,
            metrics['valid_mask']
        )

        # 4. Update
        flat_seqs = generated_seqs.reshape(-1, seq_len)
        decoder_start = torch.zeros(batch_size * self.num_samples, 1, dtype=torch.long, device=device)
        decoder_input = torch.cat([decoder_start, flat_seqs[:, :-1]], dim=1)

        expanded_input_ids = batch['input_ids'].unsqueeze(1).repeat(1, self.num_samples, 1).reshape(
            -1, batch['input_ids'].shape[-1])
        expanded_attention_mask = batch['attention_mask'].unsqueeze(1).repeat(1, self.num_samples, 1).reshape(
            -1, batch['attention_mask'].shape[-1])

        batch_expanded = {
            'input_ids': expanded_input_ids.contiguous(),
            'attention_mask': expanded_attention_mask.contiguous(),
            'decoder_input_ids': decoder_input.contiguous(),
            'labels': flat_seqs.contiguous()
        }

        self.model.eval()
        outputs = self.model(batch_expanded)
        logits = outputs.logits
        self.model.train()

        log_probs = F.log_softmax(logits.float(), dim=-1)
        token_log_probs = torch.gather(log_probs, dim=2, index=flat_seqs.unsqueeze(-1)).squeeze(-1)

        pad_mask = (flat_seqs != 0).float()
        token_log_probs = token_log_probs * pad_mask

        # 5. Reference model
        with torch.no_grad():
            self.ref_model.eval()
            ref_outputs = self.ref_model(batch_expanded)
            ref_log_probs = F.log_softmax(ref_outputs.logits.float(), dim=-1)
            ref_token_log_probs = torch.gather(ref_log_probs, dim=2, index=flat_seqs.unsqueeze(-1)).squeeze(-1) * pad_mask

        # 6. Log prob
        item_log_probs, first_eos_log_probs, item_valid_mask = self.compute_position_log_probs(
            token_log_probs, structures, seq_len, batch_size, self.num_samples
        )
        ref_item_log_probs, ref_first_eos_log_probs, _ = self.compute_position_log_probs(
            ref_token_log_probs, structures, seq_len, batch_size, self.num_samples
        )

        # 7. Item & EOS advantage
        max_items = seq_len // self.tokens_per_item
        max_positions = self.max_proactive_length
        
        # Item advantages
        item_advantages = position_advantages[:, :, :max_items]
        
        # EOS advantages
        num_items = structures['num_items']
        eos_positions = num_items.clamp(max=max_positions - 1)
        
        eos_advantages = position_advantages.gather(
            dim=-1,
            index=eos_positions.unsqueeze(-1)
        ).squeeze(-1)

        # 8. Loss caculation
        valid_mask = structures['valid']
        has_eos = structures['has_eos']
        eos_pos = structures['eos_pos']

        # Item loss
        item_policy_loss = -(item_log_probs * item_advantages * item_valid_mask.float()).sum()
        item_kl_loss = (self.beta_kl * (item_log_probs - ref_item_log_probs) * item_valid_mask.float()).sum()

        # EOS loss
        eos_valid = valid_mask & has_eos & (eos_pos < seq_len)
        eos_policy_loss = -(first_eos_log_probs * eos_advantages * eos_valid.float()).sum()
        eos_kl_loss = (self.beta_kl * (first_eos_log_probs - ref_first_eos_log_probs) * eos_valid.float()).sum()

        num_valid = valid_mask.sum().item()
        total_count = max(num_valid, 1)

        policy_loss = (item_policy_loss + eos_policy_loss) / total_count
        kl_loss = (item_kl_loss + eos_kl_loss) / total_count

        total_loss = policy_loss + kl_loss

        # 9. Collect
        stats = self._compute_stats(metrics, structures, policy_loss, kl_loss, total_loss,
                                    item_advantages, eos_advantages, position_advantages)

        return total_loss, stats

    def _compute_stats(self, metrics, structures, policy_loss, kl_loss, total_loss,
                       item_advantages, eos_advantages, position_advantages):
        item_ioi = metrics['item_ioi']
        item_ior = metrics['item_ior']
        item_ctr = metrics['item_ctr']
        fused_rewards = metrics['fused_rewards']
        sequence_ioi = metrics['sequence_ioi']
        sequence_ior = metrics['sequence_ior']
        sequence_ctr = metrics['sequence_ctr']
        sequence_fused = metrics['sequence_fused']
        valid_mask = metrics['valid_mask']
        duplicate_masks = metrics['duplicate_masks']
        valid_item_counts = metrics['valid_item_counts']
        items_mask = metrics['items_mask']
        norm_stats = metrics['norm_stats']
        real_item_mask = metrics['real_item_mask']

        valid_item_mask = (item_ctr > 0) & items_mask

        if valid_item_mask.any():
            item_ioi_mean = item_ioi[valid_item_mask].mean().item()
            item_ior_mean = item_ior[valid_item_mask].mean().item()
            item_ctr_mean = item_ctr[valid_item_mask].mean().item()
            fused_reward_mean = fused_rewards[valid_item_mask].mean().item()
        else:
            item_ioi_mean = item_ior_mean = item_ctr_mean = fused_reward_mean = 0.0

        valid_ratio = valid_mask.float().mean().item()

        if valid_mask.any():
            seq_ioi_mean = sequence_ioi[valid_mask].mean().item()
            seq_ior_mean = sequence_ior[valid_mask].mean().item()
            seq_ctr_mean = sequence_ctr[valid_mask].mean().item()
            seq_fused_mean = sequence_fused[valid_mask].mean().item()
        else:
            seq_ioi_mean = seq_ior_mean = seq_ctr_mean = seq_fused_mean = 0.0

        num_duplicates = duplicate_masks.sum().item()
        total_items = valid_item_mask.sum().item()
        duplicate_ratio = num_duplicates / total_items if total_items > 0 else 0.0

        num_items = structures['num_items']
        has_eos = structures['has_eos']
        
        lengths = num_items[valid_mask].float()
        avg_total_len = lengths.mean().item() if len(lengths) > 0 else 0.0

        if valid_mask.any():
            avg_valid_len = valid_item_counts[valid_mask].mean().item()
            valid_item_ratio = (valid_item_counts[valid_mask].sum() / num_items[valid_mask].sum()).item() if num_items[valid_mask].sum() > 0 else 0.0
        else:
            avg_valid_len = 0.0
            valid_item_ratio = 0.0

        eos_ratio = has_eos[valid_mask].float().mean().item() if valid_mask.any() else 0.0

        if real_item_mask.any():
            raw_ioi_sum = item_ioi[real_item_mask].sum().item()
            raw_ior_sum = item_ior[real_item_mask].sum().item()
            raw_ctr_sum = item_ctr[real_item_mask].sum().item()
            raw_reward_count = real_item_mask.sum().item()
        else:
            raw_ioi_sum = raw_ior_sum = raw_ctr_sum = 0.0
            raw_reward_count = 0

        result = {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'kl_loss': kl_loss.item(),
            's_ioi': seq_ioi_mean,
            's_ior': seq_ior_mean,
            's_ctr': seq_ctr_mean,
            's_fused': seq_fused_mean,
            'i_ioi': item_ioi_mean,
            'i_ior': item_ior_mean,
            'i_ctr': item_ctr_mean,
            'i_fused': fused_reward_mean,
            'total_len': avg_total_len,
            'valid_len': avg_valid_len,
            'eos_ratio': eos_ratio,
            'dup': duplicate_ratio,
            'valid': valid_ratio,
            'valid_item': valid_item_ratio,
        }

        return result

    # ==================== Training ====================
    def fit(self, train_dataloader, val_dataloader, epochs, epoch_bias=0):

        train_dataloader = self.train_dataloader
        val_dataloader = self.accelerator.prepare(val_dataloader)
        should_stop = False

        tensorboard_dir = os.path.join(self.config['ckpt_dir'], 'tensorboard')
        os.makedirs(tensorboard_dir, exist_ok=True)

        if self.accelerator.is_main_process:
            self.writer = SummaryWriter(tensorboard_dir)
            self.log(f"\n{'=' * 80}")
            self.log(f"🚀 ProRL Trainer")
            self.log(f"{'=' * 80}")

            self.log(f"\n🎯 Parameters:")
            self.log(f"    max_proactive_length:  {self.max_proactive_length}")
            self.log(f"    gamma (Discount Factors):      {self.gamma}")
            self.log(f"    reward_weight_ctr:     {self.reward_weight_ctr}")
            self.log(f"    reward_weight_ioi:     {self.reward_weight_ioi}")
            self.log(f"    reward_weight_ior:     {self.reward_weight_ior}")
            self.log(f"    beta_kl:               {self.beta_kl}")
            self.log(f"    num_processes (GPUs):  {self.accelerator.num_processes}")
            self.log(f"    steps_per_epoch:       {self.steps_per_epoch}")
            self.log(f"    total_steps:           {self.total_steps}")
            self.log(f"    warmup_steps:          {self.num_warmup_steps} ({100*self.num_warmup_steps/self.total_steps:.1f}%)")

            self.log(f"{'=' * 80}\n")

        global_step = 0
        for epoch in range(epoch_bias, epochs + epoch_bias):

            self.model.train()
            self.ref_model.eval()

            epoch_stats = defaultdict(list)

            pbar = tqdm(
                train_dataloader,
                desc=f"Epoch {epoch + 1}/{epochs + epoch_bias}",
                disable=not self.accelerator.is_main_process
            )

            for batch_idx, batch in enumerate(pbar):
                self.optimizer.zero_grad()
                loss, stats = self.compute_policy_loss(batch)
                self.accelerator.backward(loss)

                if self.config.get('max_grad_norm') is not None:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.config['max_grad_norm'])

                self.optimizer.step()
                self.scheduler.step()

                for k, v in stats.items():
                    if isinstance(v, (int, float)):
                        epoch_stats[k].append(v)

                pbar.set_postfix(
                    loss=f"{stats['total_loss']:.4f}",
                    len=f"{stats['total_len']:.1f}",
                    eos=f"{stats['eos_ratio']:.0%}",
                    fused=f"{stats['s_fused']:.3f}"
                )

                if self.accelerator.is_main_process and batch_idx % 10 == 0:
                    for k, v in stats.items():
                        if isinstance(v, (int, float)):
                            self.writer.add_scalar(f'train/{k}', v, global_step)
                    self.writer.flush()

                global_step += 1

            if epoch == epoch_bias and not self.reward_normalizer.is_frozen:
                self.reward_normalizer.freeze()
            
            if epoch % 10 == 0 or epoch == (epochs + epoch_bias - 1):

                test_results, all_results = self.evaluate_all_tokenizer(val_dataloader, store=False)

                if self.accelerator.is_main_process:
                    for key in test_results:
                        self.accelerator.log({f'Test_Metric/{key}': test_results[key]})

                    for i, results in enumerate(all_results):
                        for key in results:
                            self.accelerator.log({f'Test_{i}_Metric/{key}': results[key]})

                self.log(f'Test Results: {test_results}')
                for i, results in enumerate(all_results):
                    self.log(f'Test Results {i}: {results}')

            # Epoch Summary
            if self.accelerator.is_main_process:
                avg_valid_ratio = np.mean(epoch_stats['valid'])

                self.log(f"\n[Epoch {epoch + 1}] Summary:")
                self.log(f"  Loss: {np.mean(epoch_stats['total_loss']):.4f}")
                self.log(f"  Seq: ioi={np.mean(epoch_stats['s_ioi']):.4f}, ior={np.mean(epoch_stats['s_ior']):.1f}, ctr={np.mean(epoch_stats['s_ctr']):.4f}, fused={np.mean(epoch_stats['s_fused']):.4f}")
                self.log(f"  Item: ioi={np.mean(epoch_stats['i_ioi']):.4f}, ior={np.mean(epoch_stats['i_ior']):.1f}, ctr={np.mean(epoch_stats['i_ctr']):.4f}, fused={np.mean(epoch_stats['i_fused']):.4f}")
                self.log(f"  Length: total={np.mean(epoch_stats['total_len']):.2f}, valid={np.mean(epoch_stats['valid_len']):.2f}, eos_ratio={np.mean(epoch_stats['eos_ratio']):.1%}")

                if avg_valid_ratio < self.min_valid_ratio:
                    self.low_valid_count += 1
                    if self.low_valid_count >= self.valid_ratio_patience:
                        self.log(f"  ⚠️ Early stop triggered!")
                        should_stop = True
                else:
                    self.low_valid_count = 0

                for k in epoch_stats.keys():
                    self.writer.add_scalar(f'epoch/{k}', np.mean(epoch_stats[k]), epoch + 1)
                self.writer.flush()

            save_interval = self.config.get('save_interval')
            if save_interval is not None and (epoch + 1) % save_interval == 0:
                if self.accelerator.is_main_process:
                    ckpt_path = os.path.join(self.config['ckpt_dir'],
                                             f'{self.config["ckpt_name"]}_epoch_{epoch + 1}.pth')
                    self.save_states(epoch=epoch, path=ckpt_path)
                self.accelerator.wait_for_everyone()

            self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            if self.best_val_score == -1:
                self.save_states(epoch=epochs + epoch_bias - 1)
            self.writer.close()

        return should_stop

    # ==================== Evaluator ====================

    def evaluate(self, dataloader, split='test'):

        self.model.eval()
        all_results = defaultdict(list)
        val_progress_bar = tqdm(
            dataloader, total=len(dataloader), desc=f"Eval - {split}",
            disable=not self.accelerator.is_main_process,
        )
        all_results_info = {"preds": [], "scores": [], "labels": []}

        for batch in val_progress_bar:
            with torch.no_grad():
                batch = {k: v.to(self.accelerator.device) for k, v in batch.items()}

                if self.config.get('use_ddp', False):
                    model_t5 = self.model.module.t5
                else:
                    model_t5 = self.model.t5

                model_t5.eval()

                max_gen_length = self.max_proactive_length * self.tokens_per_item + 1

                out = model_t5.generate(
                    batch['input_ids'],
                    num_beams=self.evaluator.maxk,
                    num_return_sequences=self.evaluator.maxk,
                    do_sample=False,
                    early_stopping=True,
                    eos_token_id=self.tokenizer.eos_token,
                    pad_token_id=0,
                    max_length=max_gen_length,
                    length_penalty=1.0,
                    return_dict_in_generate=True,
                    output_scores=True
                )

                batch_size = batch['input_ids'].shape[0]
                preds = out.sequences[:, 1:].view(batch_size, self.evaluator.maxk, -1)

                # ========== safe pad for gather ==========
                preds_padded = self.accelerator.pad_across_processes(
                    preds,
                    dim=2,
                    pad_index=0,
                    pad_first=False
                )

                if self.config.get('use_ddp', False):
                    all_preds, all_input_ids, all_labels, all_proact_seman_id = self.accelerator.gather_for_metrics(
                        (preds_padded, batch['input_ids'], batch['labels'], batch['proact_seman_id']))
                    all_preds = self.zero_after_eos(all_preds, target_value=self.tokenizer.eos_token)
                    results = self.evaluator.calculate_metrics(all_preds, all_labels, all_proact_seman_id,
                                                               all_input_ids)
                    all_results_info["preds"].append(all_preds.detach().cpu())
                    all_results_info["labels"].append(all_labels.detach().cpu())
                else:
                    preds = self.zero_after_eos(preds, target_value=self.tokenizer.eos_token)
                    results = self.evaluator.calculate_metrics(preds, batch['labels'], batch['proact_seman_id'],
                                                               batch['input_ids'])
                    all_results_info["preds"].append(preds.detach().cpu())
                    all_results_info["labels"].append(batch['labels'].detach().cpu())

                for key, value in results.items():
                    all_results[key].append(value)

        output_results = OrderedDict()
        for metric in self.config['metrics']:
            for k in self.config['topk']:
                key = f"{metric}@{k}"
                if "mse" not in key:
                    output_results[key] = torch.cat(all_results[key]).mean().item()
                else:
                    output_results[key] = torch.mean(torch.tensor(all_results[key]))

        return output_results, all_results_info

    def evaluate_all_tokenizer(self, dataloader, split='test', store=False):
        tokenizer_num = dataloader.collate_fn.tokenizer_num
        results_list = []

        for tokenizer_id in range(tokenizer_num):
            dataloader.collate_fn.set_tokenizer(tokenizer_id)
            results, results_info = self.evaluate(dataloader, split)
            results_list.append(results)

        mean_results = OrderedDict()
        for metric in self.config['metrics']:
            for k in self.config['topk']:
                key = f"{metric}@{k}"
                mean_results[key] = np.mean([result[key] for result in results_list])

        return mean_results, results_list

    # ==================== Save / Load ====================

    def save_states(self, epoch=0, path=None):
        path = path or self.saved_model_ckpt

        if self.accelerator.is_main_process:
            if self.config.get('use_ddp', False):
                unwrapped_model = self.accelerator.unwrap_model(self.model)
                unwrapped_optimizer = self.accelerator.unwrap_model(self.optimizer)
                unwrapped_scheduler = self.accelerator.unwrap_model(self.scheduler)
                states = {
                    'model': unwrapped_model.state_dict(),
                    'optimizer': unwrapped_optimizer.state_dict(),
                    'scheduler': unwrapped_scheduler.state_dict(),
                }
            else:
                states = {
                    'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                }

            states.update({
                'epoch': epoch,
                'best_val_score': self.best_val_score,
                'reward_normalizer_state': self.reward_normalizer.get_state(),
            })
            torch.save(states, path)
            self.log(f"[Epoch {epoch + 1}] Saved checkpoint to {path}")

    def load_states(self, ckpt_path=None):
        ckpt_path = ckpt_path or self.saved_model_ckpt
        ckpt = torch.load(ckpt_path, map_location='cpu')
        self.log(f"Loading checkpoint from {ckpt_path}")

        if self.config.get('use_ddp', False):
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_optimizer = self.accelerator.unwrap_model(self.optimizer)
            unwrapped_scheduler = self.accelerator.unwrap_model(self.scheduler)
            unwrapped_model.load_state_dict(ckpt['model'])
            unwrapped_optimizer.load_state_dict(ckpt['optimizer'])
            unwrapped_scheduler.load_state_dict(ckpt['scheduler'])
            self.model, self.optimizer, self.scheduler = self.accelerator.prepare(
                unwrapped_model, unwrapped_optimizer, unwrapped_scheduler
            )
        else:
            self.model.load_state_dict(ckpt['model'])
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.scheduler.load_state_dict(ckpt['scheduler'])

        if 'best_val_score' in ckpt:
            self.best_val_score = ckpt['best_val_score']
        if 'epoch' in ckpt:
            self.best_epoch = ckpt['epoch']
        if 'reward_normalizer_state' in ckpt:
            self.reward_normalizer.load_state(ckpt['reward_normalizer_state'])

    def log(self, message, level='info'):
        return log(message, self.accelerator, self.logger, level=level)

    def end(self):
        self.accelerator.end_training()

# ==================== Reward Caculator ====================
class OptimizedRewardComputer_NoCOT:

    def __init__(self, evaluator, tokenizer, config):
        self.evaluator = evaluator
        self.tokenizer = tokenizer
        self.config = config
        self.tokens_per_item = 4

        self.sasrec_model = evaluator.sasrec_model
        self.sasrec_idmap = evaluator.sasrec_idmap
        self.sas_config = evaluator.sas_config
        self.item_id_name = evaluator.item_id_name

    def compute_rewards_batch(self, items_tensor, items_mask, valid_mask,
                              num_items, proact_seman_ids, input_ids):
        """batch IoI/IoR/CTR"""
        batch_size, num_samples, max_items, tokens_per_item = items_tensor.shape
        device = self.config['device']
        max_len = self.sas_config['MAX_ITEM_LIST_LENGTH']

        all_seq_before = []
        all_seq_after = []
        all_seq_lengths_before = []
        all_seq_lengths_after = []
        all_target_sas_ids = []
        all_proactive_sas_ids = []
        metadata = []

        num_items_cpu = num_items.cpu().numpy()
        valid_mask_cpu = valid_mask.cpu().numpy()

        for b in range(batch_size):
            history_raw, target_raw = self._convert_history_target(
                input_ids[b], proact_seman_ids[b]
            )

            if target_raw not in self.sasrec_idmap:
                continue
            target_sas_id = self.sasrec_idmap[target_raw]

            for s in range(num_samples):
                if not valid_mask_cpu[b, s]:
                    continue

                n = int(num_items_cpu[b, s])
                if n == 0:
                    continue

                items_raw = self._convert_items(items_tensor[b, s, :n])

                valid_prefix = []
                for k in range(n):
                    if items_raw[k] is None:
                        continue

                    seq_before = history_raw + valid_prefix
                    mapped_before, len_before = self._map_and_pad(seq_before, max_len)

                    valid_prefix.append(items_raw[k])
                    seq_after = history_raw + valid_prefix.copy()
                    mapped_after, len_after = self._map_and_pad(seq_after, max_len)

                    all_seq_before.append(mapped_before)
                    all_seq_after.append(mapped_after)
                    all_seq_lengths_before.append(len_before)
                    all_seq_lengths_after.append(len_after)
                    all_target_sas_ids.append(target_sas_id)
                    all_proactive_sas_ids.append(self.sasrec_idmap[items_raw[k]])
                    metadata.append((b, s, k))

        total_items = len(metadata)

        if total_items == 0:
            zeros = torch.zeros(batch_size, num_samples, max_items, device=device)
            return zeros.clone(), zeros.clone(), zeros.clone()

        total_seqs = total_items * 2

        interleaved_seqs = []
        interleaved_lens = []
        for i in range(total_items):
            interleaved_seqs.append(all_seq_before[i])
            interleaved_lens.append(all_seq_lengths_before[i])
            interleaved_seqs.append(all_seq_after[i])
            interleaved_lens.append(all_seq_lengths_after[i])

        seq_tensor = torch.tensor(interleaved_seqs, dtype=torch.long, device=device)
        len_tensor = torch.tensor(interleaved_lens, dtype=torch.long, device=device)
        target_tensor = torch.tensor(all_target_sas_ids, dtype=torch.long, device=device)
        proactive_tensor = torch.tensor(all_proactive_sas_ids, dtype=torch.long, device=device)

        id_name = f"{self.item_id_name}_list"

        with torch.no_grad():
            self.sasrec_model.eval()

            scores = self.sasrec_model.full_sort_predict({
                id_name: seq_tensor,
                'item_length': len_tensor
            })
            probs = torch.softmax(scores, dim=-1)

            before_indices = torch.arange(0, total_seqs, 2, device=device)
            user_embeddings = self.sasrec_model.forward(
                seq_tensor[before_indices],
                len_tensor[before_indices]
            )
            item_embeddings = self.sasrec_model.item_embedding.weight[proactive_tensor]

        prob_before = probs[before_indices]
        prob_after = probs[before_indices + 1]

        idx_range = torch.arange(total_items, device=device)
        target_prob_before = prob_before[idx_range, target_tensor]
        target_prob_after = prob_after[idx_range, target_tensor]

        ioi_values = torch.log(target_prob_after + 1e-10) - torch.log(target_prob_before + 1e-10)

        rank_before = (prob_before > target_prob_before.unsqueeze(1)).sum(dim=1) + 1
        rank_after = (prob_after > target_prob_after.unsqueeze(1)).sum(dim=1) + 1
        ior_values = (rank_before - rank_after).float()

        ctr_values = torch.sigmoid((user_embeddings * item_embeddings).sum(dim=1))

        item_ioi = torch.zeros(batch_size, num_samples, max_items, device=device)
        item_ior = torch.zeros(batch_size, num_samples, max_items, device=device)
        item_ctr = torch.zeros(batch_size, num_samples, max_items, device=device)

        if len(metadata) > 0:
            b_indices = torch.tensor([m[0] for m in metadata], device=device, dtype=torch.long)
            s_indices = torch.tensor([m[1] for m in metadata], device=device, dtype=torch.long)
            k_indices = torch.tensor([m[2] for m in metadata], device=device, dtype=torch.long)
            
            item_ioi[b_indices, s_indices, k_indices] = ioi_values
            item_ior[b_indices, s_indices, k_indices] = ior_values
            item_ctr[b_indices, s_indices, k_indices] = ctr_values

        return item_ioi, item_ior, item_ctr

    def _convert_history_target(self, input_ids, target_seman_id):
        input_ids_list = input_ids.tolist()
        semantic_id_items = [input_ids_list[i * 4:(i + 1) * 4] for i in range(10)]
        raw_id_items = [self.tokenizer._tokens2item(s) for s in semantic_id_items]
        target_raw_id = self.tokenizer._tokens2item(target_seman_id.tolist())
        return raw_id_items, target_raw_id

    def _convert_items(self, items_semantic_tensor):
        n = items_semantic_tensor.shape[0]
        all_sem_ids = items_semantic_tensor.tolist()
        items_raw = []
        for k in range(n):
            sem_id = all_sem_ids[k]
            raw_id = self.tokenizer._tokens2item(sem_id)
            if raw_id != "None" and str(raw_id) in self.sasrec_idmap:
                items_raw.append(raw_id)
            else:
                items_raw.append(None)
        return items_raw

    def _map_and_pad(self, seq, max_len):
        mapped = []
        for raw_id in seq[-max_len:]:
            if raw_id in self.sasrec_idmap:
                mapped.append(self.sasrec_idmap[raw_id])
            else:
                continue

        actual_len = len(mapped)
        if actual_len < max_len:
            mapped = [0] * (max_len - actual_len) + mapped

        return mapped[:max_len], min(actual_len, max_len)