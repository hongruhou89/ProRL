import torch
import json
import numpy as np
import time
from collections import Counter
from functools import lru_cache

from recbole.config import Config
from recbole.data.utils import create_dataset, data_preparation
from recbole.model.sequential_recommender import SASRec, GRU4Rec, LightSANs


# ==================== Evaluator Registration ====================
EVALUATOR_MODEL_REGISTRY = {
    'SASRec': SASRec,
    'GRU4Rec': GRU4Rec
}


class Evaluator:
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.metric2func = {
            'recall': self.recall_at_k,
            'ndcg': self.ndcg_at_k
        }

        self.eos_token = self.tokenizer.eos_token
        self.maxk = max(config['topk'])

    def calculate_pos_index(self, preds, labels):
        preds = preds.detach().cpu()
        labels = labels.detach().cpu()
        assert preds.shape[1] == self.maxk, f"preds.shape[1] = {preds.shape[1]} != {self.maxk}"

        pos_index = torch.zeros((preds.shape[0], self.maxk), dtype=torch.bool)
        for i in range(preds.shape[0]):
            cur_label = labels[i].tolist()
            if self.eos_token in cur_label:
                eos_pos = cur_label.index(self.eos_token)
                cur_label = cur_label[:eos_pos]
            for j in range(self.maxk):
                cur_pred = preds[i, j].tolist()
                if cur_pred == cur_label:
                    pos_index[i, j] = True
                    break
        return pos_index

    def recall_at_k(self, pos_index, k):
        return pos_index[:, :k].sum(dim=1).cpu().float()

    def ndcg_at_k(self, pos_index, k):
        ranks = torch.arange(1, pos_index.shape[-1] + 1).to(pos_index.device)
        dcg = 1.0 / torch.log2(ranks + 1)
        dcg = torch.where(pos_index, dcg, 0)
        return dcg[:, :k].sum(dim=1).cpu().float()

    def calculate_metrics(self, preds, labels):
        results = {}
        pos_index = self.calculate_pos_index(preds, labels)
        for metric in self.config['metrics']:
            for k in self.config['topk']:
                results[f"{metric}@{k}"] = self.metric2func[metric](pos_index, k)
        return results


class PRAEvaluator:
    
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        
        self.metric2func = {
            'coherence_max': self.coherence_at_k_max,
            'IoI_max': self.IoI_at_k_max,
            'IoR_max': self.IoR_at_k_max,
            'ctr_max': self.ctr_at_k_max,
            'error': self.error_at_k
        }

        self.eos_token = self.tokenizer.eos_token
        self.maxk = max(config['topk'])
        self._embedding_cache = {}

        dname_dict = {"ml-1m": "ml-1m-sas", "Steam": "steam-merged", "Books": "amazon-books"}
        item_id_dict = {"ml-1m": "item_id", "Steam": "product_id", "Books": "item_id"}
        dname = dname_dict[config["dataset"]]
        self.item_id_name = item_id_dict[config["dataset"]]

        # Reward model Registration
        self.sas_config = Config(model='SASRec',
                                 dataset=dname,
                                 config_file_list=['config/{}_sasrec_config.yaml'.format(dname)])

        self.sas_dataset = create_dataset(self.sas_config)
        sas_train_data, _, _ = data_preparation(self.sas_config, self.sas_dataset)

        self.sasrec_model = SASRec(self.sas_config, sas_train_data.dataset).to(config['device'])
        sasrec_checkpoint_file = self.sas_config[
                                     'checkpoint_dir'] + f'/{self.sas_config["model"]}-{self.sas_config["dataset"]}.pth'

        sasrec_checkpoint = torch.load(sasrec_checkpoint_file, map_location=config['device'], weights_only=False)
        self.sasrec_model.load_state_dict(sasrec_checkpoint['state_dict'])
        self.sasrec_model.eval()
        self.sasrec_idmap = sas_train_data.dataset.field2token_id[self.item_id_name]

        # Evaluator Registration
        self.evaluator_model_name = config.get('evaluator_model', 'SASRec')
        if self.evaluator_model_name not in EVALUATOR_MODEL_REGISTRY:
            raise ValueError(
                f"evaluator_model must be one of {list(EVALUATOR_MODEL_REGISTRY.keys())}, "
                f"got '{self.evaluator_model_name}'."
            )
        eval_model_cls = EVALUATOR_MODEL_REGISTRY[self.evaluator_model_name]
        eval_model_tag = self.evaluator_model_name.lower()  # 'sasrec' / 'gru4rec'
        print("Evaluator used: ", self.evaluator_model_name)

        self.evaluator_config = Config(
            model=self.evaluator_model_name,
            dataset=dname,
            config_file_list=[f'config/{dname}_{eval_model_tag}_config.yaml']
        )
        self.evaluator_dataset = create_dataset(self.evaluator_config)
        eval_train_data, _, _ = data_preparation(self.evaluator_config, self.evaluator_dataset)

        self.evaluator_model = eval_model_cls(self.evaluator_config, eval_train_data.dataset).to(config['device'])
        evaluator_ckpt_file = self.evaluator_config['checkpoint_dir'] + \
            f'/{self.evaluator_config["model"]}-{self.evaluator_config["dataset"]}.pth'
        evaluator_ckpt = torch.load(evaluator_ckpt_file, map_location=config['device'], weights_only=False)
        self.evaluator_model.load_state_dict(evaluator_ckpt['state_dict'])
        self.evaluator_model.eval()
        self.evaluator_idmap = eval_train_data.dataset.field2token_id[self.item_id_name]
        self.evaluator_max_len = self.evaluator_config['MAX_ITEM_LIST_LENGTH']

        self._tokens2item_cache = {}
        self._attribute_cache = {}
        self.max_len = self.evaluator_max_len
        self.device = config['device']

        self._evaluator_idmap_cache = {}
        self._id_name_list = "{}_list".format(self.item_id_name)

    def _cached_tokens2item(self, tokens_tuple):
        if tokens_tuple not in self._tokens2item_cache:
            self._tokens2item_cache[tokens_tuple] = self.tokenizer._tokens2item(list(tokens_tuple))
        return self._tokens2item_cache[tokens_tuple]

    def _cached_sem_ids_to_attribute(self, tokens_tuple):
        if tokens_tuple not in self._attribute_cache:
            self._attribute_cache[tokens_tuple] = self.tokenizer._sem_ids_to_attribute(list(tokens_tuple))
        return self._attribute_cache[tokens_tuple]

    def _cached_sasrec_idmap(self, raw_id):
        if raw_id not in self._evaluator_idmap_cache:
            self._evaluator_idmap_cache[raw_id] = self.evaluator_idmap[raw_id]
        return self._evaluator_idmap_cache[raw_id]

    def effective_item_extract(self, item_list):
        """
            Input: Semantic id sequence
            Output: Item list and number
        """
        len_effective_item = (item_list.index(self.eos_token) // 4) if self.eos_token in item_list else len(item_list) // 4
        item_label = [item_list[_*4:(_+1)*4] for _ in range(len_effective_item)]
        return item_label, len_effective_item

    def coherence_calculate(self, item_list, guided_item):
        itemp = item_list.copy()
        itemp.append(guided_item.tolist())

        attr_list = [self._cached_sem_ids_to_attribute(tuple(iid_tokens)) for iid_tokens in itemp]
        coherence_list = [1 if set(attr_list[i]) & set(attr_list[i + 1]) else 0
                          for i in range(len(attr_list) - 1)]
        return np.mean(coherence_list) if len(coherence_list) > 0 else 0

    def semanticid_input2rawid(self, input_id_list, semantic_target):
        semantic_id_items = [input_id_list[_ * 4:(_ + 1) * 4] for _ in range(10)]

        raw_id_items = []
        for _semantic_id in semantic_id_items:
            raw_id = self._cached_tokens2item(tuple(_semantic_id.tolist()))
            raw_id_items.append(raw_id)

        target_raw_id = self._cached_tokens2item(tuple(semantic_target.tolist()))
        return raw_id_items, target_raw_id

    def calculate_pos_index_proactive_batch_ctr_accelerate(self, preds, labels, proact_seman_id, input_ids):

        preds_cpu = preds.detach().cpu()
        labels_cpu = labels.detach().cpu()

        assert preds_cpu.shape[1] == self.maxk, f"preds.shape[1] = {preds_cpu.shape[1]} != {self.maxk}"

        batch_size = preds_cpu.shape[0]
        max_len = self.max_len
        id_mapping = self.tokenizer.id_mapping['raw_item_id2attribute']

        acc_index = torch.zeros((batch_size, self.maxk), dtype=torch.float)
        hit_index = torch.zeros((batch_size, self.maxk), dtype=torch.float)
        len_index = torch.zeros((batch_size, self.maxk), dtype=torch.float)
        len_gt_index = torch.zeros((batch_size, self.maxk), dtype=torch.float)
        error_index = torch.zeros((batch_size, self.maxk), dtype=torch.float)
        coherence_index = torch.zeros((batch_size, self.maxk), dtype=torch.float)
        IoI_index = torch.zeros((batch_size, self.maxk), dtype=torch.float)
        IoR_index = torch.zeros((batch_size, self.maxk), dtype=torch.float)
        ctr_index = torch.zeros((batch_size, self.maxk), dtype=torch.float)

        actual_len_list = []
        valid_len_list = []

        # ==================== Data collection ====================
        
        # IoI/IoR
        ioi_batch_seqs = []      
        ioi_batch_lengths = []   
        ioi_batch_meta = []      # (sample_idx, beam_idx, is_history)
        ioi_target_ids = []      
        
        # CTR
        ctr_batch_seqs = []
        ctr_batch_lengths = []
        ctr_batch_targets = []
        ctr_batch_meta = []      # (sample_idx, beam_idx, start_idx, end_idx)
        
        sample_beam_info = []    # [(sample_idx, [(beam_idx, valid_items), ...])]
        
        # Preprocess
        for i in range(batch_size):
            input_id_list = input_ids[i]
            proact_sem_id = proact_seman_id[i]
            
            # Convert history & target
            history_raw_id, target_raw_id = self.semanticid_input2rawid(input_id_list, proact_sem_id)
            target_sas_id = self._cached_sasrec_idmap(target_raw_id)
            
            # id mapping
            history_mapped = [self._cached_sasrec_idmap(rid) for rid in history_raw_id]
            
            history_for_sas = history_mapped[-max_len:] if len(history_mapped) > max_len else history_mapped
            ioi_batch_seqs.append(history_for_sas + [0] * (max_len - len(history_for_sas)))
            ioi_batch_lengths.append(len(history_raw_id))
            ioi_batch_meta.append((i, -1, True))
            ioi_target_ids.append(target_sas_id)
            
            valid_beams = []
            
            for j in range(self.maxk):
                cur_pred = preds_cpu[i, j].tolist()
                items_pred, len_pred = self.effective_item_extract(cur_pred)
                
                actual_len_list.append(len_pred)
                
                valid_item = []
                valid_item_mapped = []
                error_num = 0
                
                for i_semantic_id in items_pred:
                    tokens_tuple = tuple(i_semantic_id)
                    raw_id = self._cached_tokens2item(tokens_tuple)
                    if raw_id == "None" or raw_id not in id_mapping:
                        error_num += 1
                    else:
                        valid_item.append(raw_id)
                        valid_item_mapped.append(self._cached_sasrec_idmap(raw_id))
                
                valid_len_list.append(len(valid_item))
                
                len_index[i, j] = len_pred
                error_index[i, j] = error_num / len_pred if len_pred != 0 else 0
                coherence_index[i, j] = self.coherence_calculate(items_pred, proact_sem_id)
                
                if len(valid_item) > 0:
                    valid_beams.append((j, valid_item, valid_item_mapped))
                    
                    full_seq_mapped = history_mapped + valid_item_mapped
                    seq_for_sas = full_seq_mapped[-max_len:] if len(full_seq_mapped) > max_len else full_seq_mapped
                    ioi_batch_seqs.append(seq_for_sas + [0] * (max_len - len(seq_for_sas)))
                    ioi_batch_lengths.append(len(full_seq_mapped))
                    ioi_batch_meta.append((i, j, False))
                    ioi_target_ids.append(target_sas_id)
                    
                    ctr_start = len(ctr_batch_seqs)
                    for k in range(len(valid_item_mapped)):
                        partial_seq = history_mapped + valid_item_mapped[:k]
                        seq_for_ctr = partial_seq[-max_len:] if len(partial_seq) > max_len else partial_seq
                        ctr_batch_seqs.append(seq_for_ctr + [0] * (max_len - len(seq_for_ctr)))
                        ctr_batch_lengths.append(min(len(partial_seq), max_len))
                        ctr_batch_targets.append(valid_item_mapped[k])
                    ctr_end = len(ctr_batch_seqs)
                    ctr_batch_meta.append((i, j, ctr_start, ctr_end))
            
            sample_beam_info.append((i, valid_beams, target_sas_id))

        # ==================== IoI and IoR inference ====================
        
        if len(ioi_batch_seqs) > 0:
            ioi_seqs_tensor = torch.tensor(ioi_batch_seqs, dtype=torch.long, device=self.device)
            ioi_lens_tensor = torch.tensor(ioi_batch_lengths, dtype=torch.long, device=self.device)
            
            interaction = {
                self._id_name_list: ioi_seqs_tensor,
                'item_length': ioi_lens_tensor,
            }
            
            with torch.no_grad():
                self.evaluator_model.eval()
                scores = self.evaluator_model.full_sort_predict(interaction)
                probs = torch.softmax(scores, dim=-1)
                
                sorted_indices = torch.argsort(probs, dim=1, descending=True)
            
            history_indices = {}
            beam_indices = {}
            
            for idx, (sample_idx, beam_idx, is_history) in enumerate(ioi_batch_meta):
                if is_history:
                    history_indices[sample_idx] = idx
                else:
                    beam_indices[(sample_idx, beam_idx)] = idx
            
            # IoI/IoR calculation
            for sample_idx, valid_beams, target_sas_id in sample_beam_info:
                if sample_idx not in history_indices:
                    continue
                    
                hist_idx = history_indices[sample_idx]
                prob_before = probs[hist_idx, target_sas_id]
                rank_before = (sorted_indices[hist_idx] == target_sas_id).nonzero(as_tuple=True)[0]
                rank_before = rank_before[0].item() + 1 if len(rank_before) > 0 else probs.shape[1]
                
                for beam_idx, _, _ in valid_beams:
                    key = (sample_idx, beam_idx)
                    if key in beam_indices:
                        b_idx = beam_indices[key]
                        prob_after = probs[b_idx, target_sas_id]
                        rank_after = (sorted_indices[b_idx] == target_sas_id).nonzero(as_tuple=True)[0]
                        rank_after = rank_after[0].item() + 1 if len(rank_after) > 0 else probs.shape[1]
                        
                        IoI_index[sample_idx, beam_idx] = (torch.log(prob_after) - torch.log(prob_before)).cpu()
                        IoR_index[sample_idx, beam_idx] = -(rank_after - rank_before)

        # ==================== CTR Inference ====================
        
        if len(ctr_batch_seqs) > 0:
            ctr_seqs_tensor = torch.tensor(ctr_batch_seqs, dtype=torch.long, device=self.device)
            ctr_lens_tensor = torch.tensor(ctr_batch_lengths, dtype=torch.long, device=self.device)
            ctr_targets_tensor = torch.tensor(ctr_batch_targets, dtype=torch.long, device=self.device)
            
            with torch.no_grad():
                user_embeddings = self.evaluator_model.forward(ctr_seqs_tensor, ctr_lens_tensor)
                item_embeddings = self.evaluator_model.item_embedding(ctr_targets_tensor)
                all_scores = torch.sigmoid((user_embeddings * item_embeddings).sum(dim=1))
            
            all_scores_cpu = all_scores.cpu()
            for sample_idx, beam_idx, start_idx, end_idx in ctr_batch_meta:
                if end_idx > start_idx:
                    ctr_index[sample_idx, beam_idx] = all_scores_cpu[start_idx:end_idx].mean()

        return acc_index, hit_index, len_index, len_gt_index, error_index, coherence_index, IoI_index, IoR_index, ctr_index

    # ==================== Metric Function ====================
    def ctr_at_k_max(self, acc_index, hit_index, len_index, len_gt_index, error_index, coherence_index, IoI_index,
                     IoR_index, ctr_index, k):
        return torch.max(ctr_index[:, :k], dim=1)[0]

    def IoI_at_k_max(self, acc_index, hit_index, len_index, len_gt_index, error_index, coherence_index, IoI_index,
                     IoR_index, ctr_index, k):
        return torch.max(IoI_index[:, :k], dim=1)[0]

    def IoR_at_k_max(self, acc_index, hit_index, len_index, len_gt_index, error_index, coherence_index, IoI_index,
                     IoR_index, ctr_index, k):
        return torch.max(IoR_index[:, :k], dim=1)[0]

    def coherence_at_k_max(self, acc_index, hit_index, len_index, len_gt_index, error_index, coherence_index,
                           IoI_index, IoR_index, ctr_index, k):
        return torch.max(coherence_index[:, :k], dim=1)[0]

    def error_at_k(self, acc_index, hit_index, len_index, len_gt_index, error_index, coherence_index, IoI_index,
                   IoR_index, ctr_index, k):
        return error_index[:, :k].sum(dim=1).cpu().float()

    def calculate_metrics(self, preds, labels, proact_seman_id, input_ids):
        results = {}
        acc_index, hit_index, len_index, len_gt_index, error_index, coherence_index, IoI_index, IoR_index, ctr_index = \
            self.calculate_pos_index_proactive_batch_ctr_accelerate(preds, labels, proact_seman_id, input_ids)
        
        for metric in self.config['metrics']:
            for k in self.config['topk']:
                results[f"{metric}@{k}"] = self.metric2func[metric](
                    acc_index, hit_index, len_index, len_gt_index,
                    error_index, coherence_index, IoI_index, IoR_index, ctr_index, k
                )
        return results