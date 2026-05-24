import copy

from torch.utils.data import ConcatDataset, DataLoader

from dataset import ProactiveRecDataset
from tokenizer import Tokenizer, PRARecTokenizer

def get_proactive_datasets(config):

    train_dataset = ProactiveRecDataset(config, split='train')
    valid_dataset = ProactiveRecDataset(config, split='val')
    test_dataset = ProactiveRecDataset(config, split='test')

    return train_dataset, valid_dataset, test_dataset

def get_tokenizers(config):
    tokenizers = []

    tokenizers.append(PRARecTokenizer(config))
    return tokenizers

def get_dataloader(config, dataset, collate_fn, split):
    
    num_workers = config.get('num_proc', 4)
    
    loader_kwargs = {
        'collate_fn': collate_fn,
        'num_workers': num_workers,
        'pin_memory': True if num_workers > 0 else False,
        'persistent_workers': True if num_workers > 0 else False,
    }

    if split == 'train':
        dataloader = DataLoader(dataset, batch_size=config['train_batch_size'], 
                                shuffle=True, **loader_kwargs)
    else:
        dataloader = DataLoader(dataset, batch_size=config['eval_batch_size'], 
                                shuffle=False, **loader_kwargs)

    return dataloader
