import random
from logging import getLogger

import torch
import copy
from utils import log
import numpy as np

class Collator_Pretrain:

    def __init__(self, config, tokenizers, type = "train"):

        self.config = config
        self.logger = getLogger()
        self.tokenizers = tokenizers
        self.tokenizer_id = None
        self.select_prob = [1/self.tokenizer_num] * self.tokenizer_num
        self.type = type

    @property
    def tokenizer_num(self):
        return len(self.tokenizers)

    def __call__(self, batch):
        all_input_ids, all_attention_mask, all_labels, all_proact_seman_id = [], [], [], []
        for example in batch:
            if self.tokenizer_id is None:
                tokenizer = np.random.choice(self.tokenizers, p=self.select_prob)
            else:
                tokenizer = self.tokenizers[self.tokenizer_id]
            d = tokenizer.tokenize(example)
            all_input_ids.append(d["input_ids"])
            all_attention_mask.append(d["attention_mask"])
            all_labels.append(d["labels"])
            all_proact_seman_id.append(d["proact_seman_id"])

        all_input_ids = torch.stack(all_input_ids, dim=0)
        all_attention_mask = torch.stack(all_attention_mask, dim=0)
        all_labels = torch.stack(all_labels, dim=0)
        all_proact_seman_id = torch.stack(all_proact_seman_id, dim=0)

        if self.type != "train":
            return {
                'input_ids': all_input_ids,
                'attention_mask': all_attention_mask,
                'labels': all_labels,
                'proact_seman_id': all_proact_seman_id  # for calculate coherence
            }
        
        else:
            return {
                'input_ids': all_input_ids,
                'attention_mask': all_attention_mask,
                'labels': all_labels
            }

    def set_tokenizer(self, tokenizer_id):
        self.tokenizer_id = tokenizer_id

    def set_select_prob(self, select_prob):
        self.select_prob = select_prob

    def tokens2item(self, tokens):
        if self.tokenizer_id is None:
            tokenizer = np.random.choice(self.tokenizers, p=self.select_prob)
        else:
            tokenizer = self.tokenizers[self.tokenizer_id]

        return tokenizer._tokens2item(tokens)

    def log(self, message, level='info'):

        return log(message, self.config['accelerator'], self.logger, level=level)
    
class Collator_RL:

    def __init__(self, config, tokenizers, type = "train"):

        self.config = config
        self.logger = getLogger()
        self.tokenizers = tokenizers
        self.tokenizer_id = None
        self.select_prob = [1/self.tokenizer_num] * self.tokenizer_num
        self.type = type

    @property
    def tokenizer_num(self):
        return len(self.tokenizers)

    def __call__(self, batch):
        all_input_ids, all_attention_mask, all_labels, all_proact_seman_id = [], [], [], []
        for example in batch:
            if self.tokenizer_id is None:
                tokenizer = np.random.choice(self.tokenizers, p=self.select_prob)
            else:
                tokenizer = self.tokenizers[self.tokenizer_id]
            d = tokenizer.tokenize(example)
            all_input_ids.append(d["input_ids"])
            all_attention_mask.append(d["attention_mask"])
            all_labels.append(d["labels"])
            all_proact_seman_id.append(d["proact_seman_id"])

        all_input_ids = torch.stack(all_input_ids, dim=0)
        all_attention_mask = torch.stack(all_attention_mask, dim=0)
        all_labels = torch.stack(all_labels, dim=0)
        all_proact_seman_id = torch.stack(all_proact_seman_id, dim=0)

        return {
                'input_ids': all_input_ids,
                'attention_mask': all_attention_mask,
                'labels': all_labels,
                'proact_seman_id': all_proact_seman_id  # for calculate coherence
            }

    def set_tokenizer(self, tokenizer_id):
        self.tokenizer_id = tokenizer_id

    def set_select_prob(self, select_prob):
        self.select_prob = select_prob

    def tokens2item(self, tokens):
        if self.tokenizer_id is None:
            tokenizer = np.random.choice(self.tokenizers, p=self.select_prob)
        else:
            tokenizer = self.tokenizers[self.tokenizer_id]

        return tokenizer._tokens2item(tokens)

    def log(self, message, level='info'):

        return log(message, self.config['accelerator'], self.logger, level=level)
