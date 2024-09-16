# -*- coding: utf-8 -*-

"""
Created on 2021/4/14
@author: Siqi Miao
"""

import os.path as osp
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import permutations, product
from .root2df import Root2Df

import torch
import torch_geometric
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader, DataListLoader
import pickle
import multiprocessing

class Tau3MuDataset(InMemoryDataset):
    def __init__(self, setting, data_config, endcap, debug=False): # Instantiate relevant variables
        self.setting = setting
        self.data_dir = Path(data_config['data_dir'])
        self.conditions = data_config.get('conditions', False)
        self.node_feature_names = data_config['node_feature_names']
        self.only_one_tau = data_config['only_one_tau']
        self.splits = data_config['splits']
        self.pos_neg_ratio = data_config['pos_neg_ratio']

        self.n_hits_min = data_config.get('n_hits_min', 1)
        self.n_hits_max = data_config.get('n_hits_max', np.inf)
        
        self.coords = data_config.get('coords')

        global signal_dataset ## TODO: Figure out a way to not require these being global variables
        global bkg_dataset
        global far_station
        
        signal_dataset = data_config['signal_dataset']
        bkg_dataset = data_config['bkg_dataset']
        far_station = data_config.get('far_station', False)
        
        
        self.eta_thresh = data_config.get('eta_thresh', False) 
        
        self.cut = data_config.get('cut', False)

        self.debug = debug

        print(f'[INFO] Debug mode: {self.debug}')
        
        print(self.setting)
        print(self.raw_file_names)
        super(Tau3MuDataset, self).__init__(root=self.data_dir)
        self.data, self.slices, self.idx_split = torch.load(self.processed_paths[endcap])
        self.x_dim = self.data.x.shape[-1]
        print_splits(self)
    
    @property
    def raw_file_names(self):
        return ['DsTau3muPU0_MTD.pkl', 'DSTau3Mu_pCut1GeV_DF.pkl', 'MinBiasPU200_MTD.pkl'] if 'mix' in self.setting else [signal_dataset, bkg_dataset]

    @property
    def processed_dir(self) -> str:
        cut_id = '-' + self.cut if self.cut else ''
        dataset_name = self.raw_file_names[0].replace('.pkl', '') + '_' + self.raw_file_names[1].replace('.pkl', '')
        return osp.join(self.root, f'processed-{self.setting}{cut_id}_{dataset_name}')

    @property
    def processed_file_names(self):
        if 'half' in self.setting:
            return ['data_pos.pt', 'data_neg.pt']
        else:
            return 'data.pt'

    def download(self):

        print('Please put .pkl or .csv files ino $PROJECT_DIR/data/raw!')
        raise KeyboardInterrupt

    def process(self):
        df = self.get_df()
        
        df = df.sample(frac=1, random_state=42).reset_index(drop=False) # Shuffle the dataset. Set seed to make results reproducible
        df = df.rename(columns={'index': 'og_index'}) # Store the indices of the original dataset for prediction analysis
        
        if self.debug:
            df = df.iloc[:100]
        
        global eta
        global phi
        global r
        global z
        global theta
        
        if 'mu_hit_global_eta' in df.keys():
            eta = 'mu_hit_global_eta'
            phi = 'mu_hit_global_phi'
            r = 'mu_hit_global_r'
            z = 'mu_hit_global_z'
            theta = 'mu_hit_global_theta'
        else:
            eta = 'mu_hit_sim_eta'
            phi = 'mu_hit_sim_phi'
            r = 'mu_hit_sim_r'
            z = 'mu_hit_sim_z'
            theta = 'mu_hit_sim_theta'
        
        if 'EMTF' in self.node_feature_names:
            assert 'full' in self.setting, 'half detector setting is currently not supported'
            pt = 'EMTF_mu_pt'
            eta = 'EMTF_mu_eta'
            phi = 'EMTF_mu_phi'
        
        self.feature_names = self.node_feature_names
        
        if 'mu_hit_sim_cosphi' in self.feature_names or 'mu_hit_sim_sinphi' in self.feature_names:
            # Add phi transformations
            r_copy = df[r].to_numpy()
            phi_copy = df[phi].to_numpy()
            cos = []
            sin = []
            x = []
            y = []
            
            for i in range(len(r_copy)):
                
                hit_r = r_copy[i]
                hit_phi = phi_copy[i]
                
                cos.append(np.cos(hit_phi))
                sin.append(np.sin(hit_phi))
                
                x.append(hit_r*np.cos(hit_phi))
                y.append(hit_r*np.sin(hit_phi))
                
            df['mu_hit_sim_cosphi'] = cos
            df['mu_hit_sim_sinphi'] = sin
            df['mu_hit_sim_x'] = x
            df['mu_hit_sim_y'] = y
        
        # TODO: Figure out how to not require these to be global.
        
        if 'half' in self.setting:
            global pos_maxs
            global pos_mins
            global neg_maxs
            global neg_mins
            
            for name in ['mu_hit_nlog_eta', 'mu_hit_nlog_phi', 'mu_hit_dist', 'mu_hit_dot']:
                if name in self.feature_names:
                    self.feature_names.remove(name)
            
            # Need separate mins/maxs for each endcap
            
            pos_maxs = [[] for feature in self.feature_names]
            pos_mins = [[] for feature in self.feature_names]
            
            neg_maxs = [[] for feature in self.feature_names]
            neg_mins = [[] for feature in self.feature_names]
            
            
            #self.feature_names = np.array(feature_names)
    
            
            for i in range(len(df)):
                event = df.iloc[i]
                for j,feature in enumerate(self.feature_names):
                    var = event[feature]
                    endcaps = np.sign(event[z])
                    
                    pos_col = var[endcaps==1]
                    neg_col = var[endcaps==-1]
                    
                    if np.sum(endcaps==1) > 0:
                        pos_maxs[j].append(np.max(pos_col))
                        pos_mins[j].append(np.min(pos_col))
                    
                    if np.sum(endcaps==-1) > 0:
                        neg_maxs[j].append(np.max(neg_col))
                        neg_mins[j].append(np.min(neg_col))
                        
                        
            pos_maxs = np.array(pos_maxs)
            pos_maxs = np.max(pos_maxs, axis=1)
            neg_maxs = np.array(neg_maxs)
            neg_maxs = np.max(neg_maxs, axis=1)
            
            pos_mins = np.array(pos_mins)
            pos_mins = np.min(pos_mins, axis=1)
            neg_mins = np.array(neg_mins)
            neg_mins = np.min(neg_mins, axis=1)
            
            pos_data_list = []
            neg_data_list = []
            data_list = [pos_data_list,neg_data_list]
            
        else:
            
            global full_maxs
            global full_mins
            
            full_maxs = []
            full_mins = []
            
            for j,feature in enumerate(self.feature_names):
                
                feat = df[feature].to_numpy()
            
                full_maxs.append(np.max(np.concatenate(feat)))
                full_mins.append(np.min(np.concatenate(feat)))
            
            data_list = []
            
        print('[INFO] Processing entries...')
        args_agg = []
    
        for entry in tqdm(df.itertuples(), total=len(df)):
            
            masked_entry = self.mask_hits(entry, self.conditions, n_hits_min=self.n_hits_min)
            if masked_entry == None: # Don't use events that don't have any hits left after the mask is applied
                continue
            
            if 'half' not in self.setting:
                #args_agg.append(masked_entry)
     
                data_list.append(self._process_one_entry(masked_entry))
            else:
                for i, endcap in enumerate([1,-1]):
                    if 'half' in self.setting:
                        entry = Tau3MuDataset.split_endcap(masked_entry, endcap)
                        
                        if entry == None: # Don't use an endcap that is empty
                            continue
                        else:
                            # half-detector, tau and non-tau endcap
                            data_list[i].append(self._process_one_entry(entry, endcap=endcap))
                    
        if 'half' in self.setting:
            for i in range(2):
                idx_split = Tau3MuDataset.get_idx_split(data_list[i], self.splits, self.pos_neg_ratio)
                data, slices = self.collate(data_list[i])
        
                print('[INFO] Saving data.pt...')
                torch.save((data, slices, idx_split), self.processed_paths[i])
        
        else:
            
            data, slices = self.collate(data_list)
            idx_split = Tau3MuDataset.get_idx_split(data_list, self.splits, self.pos_neg_ratio)
            print(self.processed_paths)
            torch.save((data, slices, idx_split), self.processed_paths[0])

    def _process_one_entry(self, entry, endcap=0, only_eval=False):
        
        if endcap == 0:
            maxs = full_maxs
            mins = full_mins
        elif endcap == 1:
            maxs = pos_maxs
            mins = pos_mins
        elif endcap == -1:
            maxs = neg_maxs
            mins = neg_mins

        if 'half' in self.setting:
            if entry['n_gen_tau']==1: # If signal event, only return hits on tau endcap
                if ((entry['gen_tau_eta'] * entry[eta]) > 0).sum() == entry['n_mu_hit']: 
                    only_eval=False
                else:
                    only_eval=True
        else:
            only_eval = False
        
        for i, feature in enumerate(self.feature_names): # Min-max norm
            entry[feature] = (entry[feature] - mins[i]) / (maxs[i] - mins[i])
        
        x = Tau3MuDataset.get_node_features(entry, self.node_feature_names)
        coords = self.get_coors_for_hits(entry)
        
        gen_mu = Tau3MuDataset.get_gen_mu_kinematics(entry)
        gen_tau = Tau3MuDataset.get_gen_tau_kinematics(entry)
        y = torch.tensor(entry['y']).float().view(-1, 1)
        
        if 'mu_hit_truth' in entry.keys():
            hit_truth = self.get_hit_truth(entry)
            return Data(x=x, y=y, coords=coords, sample_idx=entry['og_index'], endcap=endcap, only_eval=only_eval, gen_mu=gen_mu, gen_tau=gen_tau, hit_truth=hit_truth)

        else:
            return Data(x=x, y=y, coords=coords, sample_idx=entry['og_index'], endcap=endcap, only_eval=only_eval, gen_mu=gen_mu, gen_tau=gen_tau)

        

        


    def get_df_save_path(self):
        save_name = ''
        save_name += 'mix_' if 'mix' in self.setting else 'raw_'
        
        save_name += self.raw_file_names[0].replace('.pkl', '') + '_' + self.raw_file_names[1].replace('.pkl', '') + '_'
        save_name += self.cut if self.cut else 'nocut'
        df_dir = self.data_dir / 'scores' / f'{save_name}'
        df_dir.mkdir(parents=True, exist_ok=True)
        return df_dir / f'{save_name}.pkl'

    def get_df(self):
        df_save_path = self.get_df_save_path()
        if df_save_path.exists():
            print(f'[INFO] Loading {df_save_path}...')
            with open(df_save_path, 'rb') as handle: 
                return pickle.load(handle)
    
        
        dfs = Root2Df(self.data_dir / 'raw').read_df(self.setting)
        neg200 = dfs[self.raw_file_names[1].replace('.pkl', '')]
        pos200 = dfs[self.raw_file_names[0].replace('.pkl', '')]
        pos0 = dfs.get('DsTau3muPU0_MTD', None)
        
        #assert self.only_one_tau # Only one-tau is supported
        if self.only_one_tau:
            pos200 = pos200[pos200.n_gen_tau == 1].reset_index(drop=True)
            if pos0 is not None:
                pos0 = pos0[pos0.n_gen_tau == 1].reset_index(drop=True)
            
            #try:
            #    neg200 = neg200[neg200.n_gen_tau == 1].reset_index(drop=True)
            #except:
            #    pass
            
        if self.cut:
            pos200 = pos200[pos200.apply(lambda x: self.filter_samples(x), axis=1)].reset_index(drop=True)
            if pos0 is not None:
                pos0 = pos0[pos0.apply(lambda x: self.filter_samples(x), axis=1)].reset_index(drop=True)

        if pos0 is not None and len(pos0) > 100000:
            print('[INFO] Sampling from pos0 to fasten processing & training...')
            pos0 = pos0.sample(100000).reset_index(drop=True)

        if 'mix' in self.setting:
            pos, neg = self.mix(pos0, neg200, pos200, self.setting)
        else:
            pos, neg = pos200, neg200
        
        if 'half' in self.setting:
            min_pos_neg_ratio = len(pos) / (len(neg) * 2)
        else:
            min_pos_neg_ratio = len(pos) / len(neg)
        print(f'[INFO] min_pos_neg_ratio: {min_pos_neg_ratio}')

        pos['y'], neg['y'] = 1, 0
        print(pos)
        print(neg)
        #assert self.pos_neg_ratio >= min_pos_neg_ratio, f'min_pos_neg_ratio = {min_pos_neg_ratio}! Now pos_neg_ratio = {self.pos_neg_ratio}!'
        
        print(f'[INFO] Concatenating pos & neg, saving to {df_save_path}...')
        df = pd.concat((pos, neg), join='outer', ignore_index=True)
        df.to_pickle(df_save_path)
        return df


    @staticmethod
    def get_node_features(entry, feature_names):

        # Directly index the entry using features = entry[feature_names] is extremely slow!
        features = np.stack([entry[feature] for feature in feature_names], axis=1)
        
        return torch.tensor(features, dtype=torch.float)

    @staticmethod
    def get_gen_mu_kinematics(entry):
        
        if entry['y'] == 1:
            
            pt = entry['gen_mu_pt']
            eta = entry['gen_mu_eta']
            phi = entry['gen_mu_phi']
            
            lead, sublead, soft = np.argsort(pt)
            
            features = [np.array([pt[i],eta[i],phi[i]]) for i in (lead,sublead,soft)]
            
            features = np.concatenate(features).reshape(1,9)
        else:
            return torch.empty(1,9)
        
        return torch.tensor(features, dtype=torch.float)
    
    @staticmethod
    def get_gen_tau_kinematics(entry):
        
        if entry['y'] == 1:
            features = np.stack([entry[feature] for feature in ['gen_tau_pt', 'gen_tau_eta', 'gen_tau_phi']], axis=1).reshape(1,3)
        else:
            return torch.empty(1,3)
        
        return torch.tensor(features, dtype=torch.float)

    @staticmethod
    def get_idx_split(data, splits, pos2neg):
        np.random.seed(42)
        assert sum(splits.values()) == 1.0
        y_dist = np.array([event.y[0][0] for event in data])
        only_eval = np.array([event.only_eval for event in data])
        
        pos_idx = np.argwhere( np.logical_and(y_dist == 1,~only_eval) ).reshape(-1)  # if is half-detector model, do not use non-tau endcap for training
        neg_idx = np.argwhere( np.logical_and(y_dist == 0,~only_eval) ).reshape(-1)
        only_eval_idx = np.argwhere(only_eval).reshape(-1)
        
        print('Number of Positive Samples: ', len(pos_idx))
        print('Number of Negative Samples: ', len(neg_idx))
        print('Minimum pos2neg:', len(pos_idx)/len(neg_idx))
        
        try:
            assert len(pos_idx) <= len(neg_idx) * pos2neg, 'The number of negative samples is not enough given the pos_neg_ratio!'
        except:
            print('Number of negative samples not enough. Using minimum possible')
            pos2neg = len(pos_idx)/len(neg_idx)
        
        n_train_pos, n_valid_pos = int(splits['train'] * len(pos_idx)), int(splits['valid'] * len(pos_idx))
        if pos2neg > 0:
            n_train_neg, n_valid_neg = int(n_train_pos / pos2neg), int(n_valid_pos / pos2neg)
        else:
            n_train_neg, n_valid_neg = int(splits['train']*len(neg_idx)), int(splits['valid']*len(neg_idx))

        pos_train_idx = pos_idx[:n_train_pos]
        pos_valid_idx = pos_idx[n_train_pos:n_train_pos + n_valid_pos]
        pos_test_idx = pos_idx[n_train_pos + n_valid_pos:]

        neg_train_idx = neg_idx[:n_train_neg]
        neg_valid_idx = neg_idx[n_train_neg:n_train_neg + n_valid_neg]
        neg_test_idx = neg_idx[n_train_neg + n_valid_neg:]

        return {'pos_train': pos_train_idx.tolist(),
                'pos_valid': pos_valid_idx.tolist(),
                'pos_test':  pos_test_idx.tolist(),
                'neg_train': neg_train_idx.tolist(),
                'neg_valid': neg_valid_idx.tolist(),
                'neg_test':  neg_test_idx.tolist()
                
                }
    
    def get_coors_for_hits(self, entry):
            
        coors = torch.tensor(np.stack([entry[feature] for feature in self.coords]).T)
        return coors
        
    def get_hit_truth(self, entry):
        if entry['y'] == 1:

            return torch.tensor(entry['mu_hit_truth'])
        else:
            return torch.zeros(entry[eta].shape)
        
        
    @staticmethod
    def mix(pos0, neg200, pos200, setting):
        neg_idx = np.arange(len(neg200))
        np.random.shuffle(neg_idx)

        # first len(pos0) neg data will be used as noise in pos0
        noise_in_pos0 = neg200.loc[neg_idx[:len(pos0)]].reset_index(drop=True)
        # remaining neg data will remain negative
        neg200 = neg200.loc[neg_idx[len(pos0):]].reset_index(drop=True)

        print('[INFO] Mixing data...')
        # mixed_pos = noise_in_pos0
        mixed_pos = []
        for idx, entry in tqdm(pos0.iterrows(), total=len(pos0)):
            for k, v in entry.items():
                if 'gen' in k:  # directly keep gen variables
                    continue
                elif isinstance(v, int):  # accumulate n_mu_hit
                    assert k == 'n_mu_hit'
                    entry['node_label'] = np.concatenate((np.zeros(noise_in_pos0.iloc[idx][k]), np.ones(v)))
                    entry[k] += noise_in_pos0.iloc[idx][k]
                else:  # concat hit features
                    assert isinstance(v, np.ndarray)
                    mixed_hits = np.concatenate((noise_in_pos0.iloc[idx][k], v))
                    entry[k] = mixed_hits
            mixed_pos.append(entry.values)
        mixed_pos = pd.DataFrame(data=mixed_pos, columns=entry.index)

        if 'check' in setting:
            return mixed_pos, pos200
        elif 'sanity' in setting:
            return noise_in_pos0, neg200
        else:
            return mixed_pos, neg200

    @staticmethod
    def split_endcap(masked_entry, endcap):
        

        entry = {}
        endcap_idx = np.sign(masked_entry[z]) == endcap

        for k, v in masked_entry.items():
            if isinstance(v, np.ndarray) and 'gen' not in k and k != 'y' and 'L1' not in k and 'EMTF' not in k:
                assert v.shape[0] == masked_entry['n_mu_hit']
                entry[k] = v[endcap_idx]
            else:
                entry[k] = v
        
        entry['n_mu_hit'] = endcap_idx.sum().item()
        
        if entry['n_mu_hit'] < 1: # Only return an entry if it has hits left
            return None
        
        if masked_entry['y'] == 1: # If signal event, only return hits on tau endcap
            if ((masked_entry['gen_tau_eta'] * entry[eta]) > 0).sum() == entry['n_mu_hit']: 
                entry['y'] = 1
            else:
                entry['y'] = 0
        
        return entry

    def mask_hits(self, entry, conditions, n_hits_min=1):
        n_mu_hit = eval(f'len(entry.{self.node_feature_names[0]})')
        
        if n_mu_hit == 0: return None
        
        if conditions == False:
            masked_entry = {'n_mu_hit': n_mu_hit}
            for k in entry._fields:
                value = getattr(entry, k)
                if isinstance(value, np.ndarray):
                    masked_entry[k] = value.reshape(-1)
                else:
                    if k != 'n_mu_hit':
                        masked_entry[k] = value
            
            return masked_entry
        
        mask = np.ones(n_mu_hit, dtype=bool)
        for k, v in conditions.items():
            k = k.split('-')[1]
            assert isinstance(getattr(entry, k), np.ndarray)
            mask *= eval('entry.' + k + v)
        
        
        n_mu_hit = mask.sum()
        masked_entry = {'n_mu_hit': n_mu_hit}
        
        if not(mask.sum() >= n_hits_min): # Only return an entry if it has hits left
            return None
        
        for k in entry._fields:
            value = getattr(entry, k)
            if isinstance(value, np.ndarray) and 'gen' not in k and k != 'y' and 'L1' not in k and k != 'n_' not in k:
                try: masked_entry[k] = value[mask].reshape(-1)
                except:
                    continue
            else:
                if k != 'n_mu_hit':
                    masked_entry[k] = value
        return masked_entry



def get_data_loaders_contrastive(setting, data_config, batch_size, endcap=1):
    
    if endcap == 1 or endcap==0:
        idx = 0
    elif endcap == -1:
        idx = 1
    
    dataset = Tau3MuDataset(setting, data_config, idx)
    print('Retrieving Data Loaders from:'+dataset.processed_paths[idx])
    train_loader = [DataLoader(dataset[dataset.idx_split['pos_train']], batch_size=batch_size, shuffle=True,drop_last=True),DataLoader(dataset[dataset.idx_split['neg_train']], batch_size=batch_size, shuffle=True,drop_last=True)]
    valid_loader = [DataLoader(dataset[dataset.idx_split['pos_valid']], batch_size=batch_size, shuffle=True,drop_last=True),DataLoader(dataset[dataset.idx_split['neg_valid']], batch_size=batch_size, shuffle=True,drop_last=True)]
    test_loader = [DataLoader(dataset[dataset.idx_split['pos_test']], batch_size=batch_size, shuffle=True,drop_last=True),DataLoader(dataset[dataset.idx_split['neg_test']], batch_size=batch_size, shuffle=True,drop_last=True)]
    return {'train': train_loader, 'valid': valid_loader, 'test': test_loader}, dataset.x_dim, dataset


def print_splits(dataset):
    def get_pos_neg_count(y):
        pos_y = (y == 1).sum()
        neg_y = (y == 0).sum()
        pos2neg_ratio = pos_y / neg_y
        return pos_y, neg_y, pos2neg_ratio

    print('[Splits]')
    for k, v in dataset.idx_split.items():
        y = dataset.data.y[v]
        pos_y, neg_y, pos2neg_ratio = get_pos_neg_count(y)
        print(f'    {k}: {len(v)}. # pos: {pos_y}, # neg: {neg_y}. Pos:Neg: {pos2neg_ratio:.3f}')