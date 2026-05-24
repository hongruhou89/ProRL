import os
from logging import getLogger

import numpy as np
from tqdm import tqdm
import json
from collections import defaultdict
import torch
import torch.nn.functional as F

from utils import *

class Tokenizer:

	def __init__(self, config):

		self.config = config
		self.logger = getLogger()

		self.expand_final = config['expand_final']

		self.id_mapping_file = os.path.join(config['data_dir'], '{}.datamaps'.format(config['dataset']))
		self.id_mapping = load_json(self.id_mapping_file)
		self.user2id = self.id_mapping['id2user']
		self.itemid2title = self.id_mapping['itemid2title']

		self.item2tokens, self.tokens2item = self._load_item2tokens()

		self.base_user_token = sum(self.n_codebook) + 1
		self.n_user_tokens = self.config['n_user_tokens']
		self.eos_token = self.base_user_token + self.n_user_tokens

	def _sem_ids_to_tokens(self, item2sem_ids: dict):
		"""
		Converts semantic IDs to tokens.

		Args:
			item2sem_ids (dict): A dictionary mapping items to their corresponding semantic IDs.

		Returns:
			dict: A dictionary mapping items to their corresponding tokens.
		"""

		offset = np.cumsum([0] + self.n_codebook)[:-1]
		sem_ids2item = {}
		for item in item2sem_ids:
			tokens = list(item2sem_ids[item])
			for digit in range(self.n_digit):
				# "+ 1" as 0 is reserved for padding
				tokens[digit] += offset[digit] + 1
				tokens[digit] = int(tokens[digit])
			item2sem_ids[item] = tuple(tokens)
			sem_ids2item[tuple(tokens)] = item
		return item2sem_ids, sem_ids2item

	def _load_item2tokens(self):
		# Load semantic IDs
		self.tokenizer_name = f'{self.config["token_prefix"]}.{self.config["token_suffix"]}'
		sem_ids_path = os.path.join(
			self.config['data_dir'],
			self.tokenizer_name
		)
		self.log(f'[TOKENIZER] Loading semantic IDs from {sem_ids_path}...')
		item2sem_ids = json.load(open(sem_ids_path, 'r'))
		item2tokens, tokens2item = self._sem_ids_to_tokens(item2sem_ids)

		return item2tokens, tokens2item

	def _sem_ids_to_attribute(self, tokens):

		if tuple(tokens) not in self.tokens2item:
			# self.log(f'[TOKENIZER] Warning: {tokens} not in tokens2item')
			return []

		iid = self.tokens2item[tuple(tokens)]

		if str(iid) not in self.id_mapping['raw_item_id2attribute']:
			# self.log(f'[TOKENIZER] Warning: {tokens} not in tokens2item')
			return []

		return self.id_mapping['raw_item_id2attribute'][str(iid)]

	@property
	def n_digit(self):
		"""
		Returns the number of digits for the tokenizer.

		The number of digits is determined by the value of `n_codebooks` in the configuration.
		"""
		if self.expand_final:
			return self.config['n_codebooks'] + 1
		else:
			return self.config['n_codebooks']

	@property
	def n_codebook(self):
		"""
		Returns the codebook size for the TIGER tokenizer.

		If `codebook_size` is a list, it returns the list as is.
		If `codebook_size` is an integer, it returns a list with `n_digit` elements,
		where each element is equal to `codebook_size`.

		Returns:
			list: The codebook size for the TIGER tokenizer.
		"""
		if isinstance(self.config['codebook_size'], list):
			return self.config['codebook_size']
		else:
			return [self.config['codebook_size']] * self.n_digit

	def _token_single_user(self, user: str) -> int:
		"""
		Tokenizes a single user.

		Args:
			user (str): The user to tokenize.

		Returns:
			int: The tokenized user ID.

		"""
		user_id = int(self.user2id[user])
		return self.base_user_token + user_id % self.n_user_tokens

	@property
	def padding_token(self):
		return 0

	def _token_single_item(self, item: str) -> int:
		"""
		Tokenizes a single item.

		Args:
			item (str): The item to be tokenized.

		Returns:
			list: The tokens corresponding to the item.
		"""
		return self.item2tokens[item]

	def _tokens2item(self, tokens: list) -> str:
		"""
		Converts a list of tokens to the corresponding item.

		Args:
			tokens (list): The tokens to be converted.

		Returns:
			str: The item corresponding to the tokens.
		"""
		if tuple(tokens) not in self.tokens2item:
			# self.log(f'[TOKENIZER] Warning: {tokens} not in tokens2item')
			return "None"

		return self.tokens2item[tuple(tokens)]

	def tokenize(self, example: dict) -> dict:

		max_item_seq_len = self.config['max_item_seq_len']

		# input_ids
		user_token = self._token_single_user(example['user'])
		input_ids = [user_token]
		for item in example['item_seq'][:-1][-max_item_seq_len:]:
			input_ids.extend(self._token_single_item(item))
		input_ids.append(self.eos_token)
		input_ids.extend([self.padding_token] * (self.max_token_seq_len - len(input_ids)))

		# attention_mask
		item_seq_len = min(len(example['item_seq'][:-1]), max_item_seq_len)
		attention_mask = [1] * (self.n_digit * item_seq_len + 2)
		attention_mask.extend([0] * (self.max_token_seq_len - len(attention_mask)))

		# labels
		labels = list(self._token_single_item(example['item_seq'][-1])) + [self.eos_token]

		return {
			'input_ids': torch.LongTensor(input_ids),
			'attention_mask': torch.FloatTensor(attention_mask),
			'labels': torch.LongTensor(labels)
		}


	@property
	def vocab_size(self) -> int:
		"""
		Returns the vocabulary size for the TIGER tokenizer.
		"""
		return self.eos_token + 1

	@property
	def max_token_seq_len(self) -> int:
		"""
		Returns the maximum token sequence length for the TIGER tokenizer.
		"""
		# +1 for user token
		return self.config['max_item_seq_len'] * self.n_digit + 2


	def log(self, message, level='info'):

		return log(message, self.config['accelerator'], self.logger, level=level)

class PRARecTokenizer(Tokenizer):

	def __init__(self, config, sem_id_epoch = None):
		self.sem_id_epoch = sem_id_epoch
		self.config = config

		super(PRARecTokenizer, self).__init__(config)

	def tokenize(self, example: dict) -> dict:

		max_item_seq_len = self.config['max_item_seq_len']

		# input_ids
		input_ids = []

		for item in example['inters']:
			input_ids.extend(self._token_single_item(item))

		input_ids.extend(self._token_single_item(example['proc_path'][-1]))	# taget item
		input_ids.append(self.eos_token)
		input_ids.extend([self.padding_token] * (self.max_token_seq_len - len(input_ids)))	# paddings

		# attention_mask
		item_seq_len = min(len(example['inters']) + 1, max_item_seq_len + 1)	# +1 for target
		attention_mask = [1] * (self.n_digit * item_seq_len + 1 + self.config["user_id_used"])	# +1 for eos
		attention_mask.extend([0] * (self.max_token_seq_len - len(attention_mask)))

		# labels
		labels = []
		for proc_item in example["proc_path"][:-1]:
			labels.extend(self._token_single_item(proc_item))
		labels.append(self.eos_token)
		labels.extend([0] * ((self.config['max_proactive_length'] - len(example["proc_path"][:-1])) * 4))	# padding

		return {
			'input_ids': torch.LongTensor(input_ids),
			'attention_mask': torch.LongTensor(attention_mask),
			'labels': torch.LongTensor(labels),
			'proact_seman_id': torch.LongTensor(self._token_single_item(example['proc_path'][-1]))
		}