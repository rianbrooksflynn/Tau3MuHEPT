import yaml
from tqdm import tqdm
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import argparse

from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--setting', type=str, help='experiment settings', default='GNN_half_dR_1')

args = parser.parse_args()

torch.set_num_threads(5)
set_seed(42)

setting = args.setting
config = yaml.safe_load(Path(f'./configs/{setting}.yml').open('r'))
print(f'[INFO] Running {setting} on cpu')

data_loaders, x_dim, dataset = get_data_loaders_contrastive(setting, config['data'], batch_size=1)