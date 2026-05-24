import datetime
import hashlib
import logging
import os
import json
from typing import Union

from accelerate.utils import set_seed
from sentence_transformers import SentenceTransformer


def ensure_dir(dir_path):

    os.makedirs(dir_path, exist_ok=True)

def config_for_log(config: dict) -> dict:
    config = config.copy()
    config.pop('device', None)
    config.pop('accelerator', None)
    for k, v in config.items():
        if isinstance(v, list):
            config[k] = str(v)
    return config



def get_total_steps(config, train_dataloader):

    if config['steps'] is not None:
        return config['steps']
    else:
        return len(train_dataloader) * config['epochs']

def init_logger(config):
    LOGROOT = config['log_dir']
    os.makedirs(LOGROOT, exist_ok=True)
    dataset_name = os.path.join(LOGROOT, config["dataset"])
    os.makedirs(dataset_name, exist_ok=True)

    logfilename = get_file_name(config, suffix='.log')
    logfilepath = os.path.join(LOGROOT, config["dataset"], logfilename)

    filefmt = "%(asctime)-15s %(levelname)s  %(message)s"
    filedatefmt = "%a %d %b %Y %H:%M:%S"
    fileformatter = logging.Formatter(filefmt, filedatefmt)

    fh = logging.FileHandler(logfilepath)
    fh.setLevel(logging.INFO)
    fh.setFormatter(fileformatter)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)

    logging.basicConfig(level=logging.INFO, handlers=[sh, fh])


def get_file_name(config: dict, suffix: str = ''):
    config_str = "".join([str(value) for key, value in config.items() if (key != 'accelerator' and key != 'device') ])
    md5 = hashlib.md5(config_str.encode(encoding="utf-8")).hexdigest()[:6]
    logfilename = "{}-{}{}".format(config['run_local_time'], md5, suffix)
    return logfilename



def load_json(file_path):

    with open(file_path, 'r') as f:
        data =  json.load(f)

    return data

def init_seed(seed, reproducibility):
    r"""init random seed for random functions in numpy, torch, cuda and cudnn
        This function is taken from https://github.com/RUCAIBox/RecBole/blob/2b6e209372a1a666fe7207e6c2a96c7c3d49b427/recbole/utils/utils.py#L188-L205

    Args:
        seed (int): random seed
        reproducibility (bool): Whether to require reproducibility
    """

    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)
    if reproducibility:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

def convert_config_dict(config: dict) -> dict:
    """
    Convert the values in a dictionary to their appropriate types.

    Args:
        config (dict): The dictionary containing the configuration values.

    Returns:
        dict: The dictionary with the converted values.

    """
    for key in config:
        v = config[key]
        if not isinstance(v, str):
            continue
        try:
            new_v = eval(v)
            if new_v is not None and not isinstance(
                new_v, (str, int, float, bool, list, dict, tuple)
            ):
                new_v = v
        except (NameError, SyntaxError, TypeError):
            if isinstance(v, str) and v.lower() in ['true', 'false']:
                new_v = (v.lower() == 'true')
            else:
                new_v = v
        config[key] = new_v
    return config

def init_device():
    """
    Set the visible devices for training. Supports multiple GPUs.

    Returns:
        torch.device: The device to use for training.

    """
    import torch
    use_ddp = True if os.environ.get("WORLD_SIZE") else False  # Check if DDP is enabled
    if torch.cuda.is_available():
        return torch.device('cuda'), use_ddp
    else:
        return torch.device('cpu'), use_ddp

def get_local_time():
    r"""Get current time

    Returns:
        str: current time
    """
    cur = datetime.datetime.now()
    cur = cur.strftime("%b-%d-%Y_%H-%M")
    return cur


def parse_command_line_args(unparsed: list[str]) -> dict:

    args = {}
    for text_arg in unparsed:
        if '=' not in text_arg:
            # print(text_arg)
            raise ValueError(f"Invalid command line argument: {text_arg}, please add '=' to separate key and value.")
        key, value = text_arg.split('=')
        key = key[len('--'):]
        try:
            value = eval(value)
        except:
            pass
        args[key] = value
    return args



def log(message, accelerator, logger, level='info'):
    if accelerator.is_main_process:
        if level == 'info':
            logger.info(message)
        elif level == 'error':
            logger.error(message)
        elif level == 'warning':
            logger.warning(message)
        elif level == 'debug':
            logger.debug(message)
        else:
            raise ValueError(f'Invalid log level: {level}')




def encode_sent_emb(config, item2meta, id_mapping, output_path):


    sent_emb_model = SentenceTransformer(
        config['sent_emb_model']
    ).to(config['device'])

    meta_sentences = [] # 1-base, meta_sentences[0] -> item_id = 1
    for i in range(1, len(id_mapping['id2item'])):
        meta_sentences.append(item2meta[id_mapping['id2item'][i]])
    sent_embs = sent_emb_model.encode(
        meta_sentences,
        convert_to_numpy=True,
        batch_size=config['sent_emb_batch_size'],
        show_progress_bar=True,
        device=config['device']
    )

    # PCA
    if config['sent_emb_pca'] > 0:
        print(f'[TOKENIZER] Applying PCA to sentence embeddings...')
        from sklearn.decomposition import PCA
        pca = PCA(n_components=config['sent_emb_pca'], whiten=True)
        sent_embs = pca.fit_transform(sent_embs)

    sent_embs.tofile(output_path)
    return sent_embs



def list_to_str(l: Union[list, str], remove_blank=False) -> str:

    ret = ''
    if isinstance(l, list):
        ret = ', '.join(map(str, l))
    else:
        ret = l
    if remove_blank:
        ret = ret.replace(' ', '')
    return ret