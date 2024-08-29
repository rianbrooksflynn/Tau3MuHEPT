import shutil
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import yaml
from datetime import datetime
from tqdm import tqdm
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

from utils import MultiClusterLoss, Writer, log_epoch, load_checkpoint, save_checkpoint, set_seed, add_cuts_to_config
from utils import get_data_loaders_contrastive as get_data_loaders

from models.get_model import get_model
#from model_efficiency import main as model_eff
#from model_efficiency import eval_one_batch, run_one_epoch, generate_roc
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
from itertools import cycle
torch.autograd.set_detect_anomaly(True)
        
class Tau3MuGNNs:

    def __init__(self, config, device, log_path, setting):
        self.config = config
        self.device = device
        self.log_path = log_path
        self.writer = Writer(log_path)
        self.log_dir=self.writer.log_dir
        self.qat = config['model_kwargs']['qat']
        self.baseline = config['model_kwargs']['baseline']
        self.lr_s = config['optimizer'].get('lr_s', False)
        self.grad_clip = config['optimizer'].get('grad_clip', False)
        
          
        if self.qat:
            from models import QDecoder as Decoder
        else:
            from models import Decoder
        
        endcap = config['model_kwargs']['endcap']
        
        self.data_loaders, x_dim, dataset = get_data_loaders(setting, config['data'], config['optimizer']['batch_size'], endcap=endcap)
        
        self.mask_frac = config['model_kwargs']['mask_frac']
        self.model = get_model(config['model_kwargs'],dataset)   
        #self.init_model(self.model)
        
        self.decoder = Decoder(config['model_kwargs']['out_dim'])
        
        self.hept_opt = torch.optim.AdamW(self.model.parameters(), lr=config['optimizer']['lr'], weight_decay=config['optimizer']['weight_decay'])
        self.dec_opt = torch.optim.AdamW(self.decoder.parameters(), lr=config['optimizer']['lr'], weight_decay=config['optimizer']['weight_decay'])
        
        if self.lr_s:
            self.max_lr = float(config['optimizer']['max_lr'])
            self.lr_s = OneCycleLR(self.hept_opt, max_lr=self.max_lr, steps_per_epoch=len(self.data_loaders['train'][0]), epochs=self.config['optimizer']['epochs']+1, anneal_strategy='linear', pct_start=0.3)
            self.lr_s_dec = OneCycleLR(self.dec_opt, max_lr=self.max_lr, steps_per_epoch=len(self.data_loaders['train'][0]), epochs=self.config['optimizer']['epochs']+1, anneal_strategy='linear', pct_start=0.3)
            
        self.criterion = MultiClusterLoss(config['optimizer'], config['model_kwargs']['out_dim'])
 
      
    
    def init_model(self, model):
        for n,l in model.named_modules():
            if isinstance(l, nn.Linear): 
                torch.nn.init.xavier_uniform_(l.weight)
        
    @torch.no_grad()
    def eval_one_batch(self, pos_batch, new_epoch=None):
        self.model.to(self.device)
        self.model.eval()
        #self.decoder.eval()
        
        #self.decoder.to(self.device)

        with torch.no_grad():
            pos_batch.to(self.device)
            
            
            pos_embeds = self.model(pos_batch.x, pos_batch.coords, pos_batch.batch, pool=False)
            
            loss, loss_dict = self.criterion(pos_embeds, pos_batch.hit_truth, train=False)

            with torch.no_grad():
                scores, labels = self.criterion.cluster(pos_embeds.cpu())
                targets = torch.where(pos_batch.hit_truth==3, 1, 0).cpu()
                
                auc = roc_auc_score(targets, scores[-1])
                loss_dict['sigbkg_auc'] = auc
            
        del pos_batch
        return loss_dict
        
    def train_one_batch(self, pos_batch, new_epoch=False):
        self.model.to(self.device)
        #self.decoder.to(self.device)

        pos_batch.to(self.device)
        
        self.model.train()
        #self.decoder.train()
        
        pos_embeds = self.model(pos_batch.x, pos_batch.coords, pos_batch.batch, pool=False)
        
        #pos_logits = self.decoder(pos_embeds.clone().detach())
        #neg_logits = self.decoder(neg_embeds.clone().detach())
        #event_clf_logits = torch.cat([pos_logits, neg_logits]).data.cpu()
        
        loss, loss_dict = self.criterion(pos_embeds, pos_batch.hit_truth)
        
        self.hept_opt.zero_grad()
        loss.backward()
        #loss.backward(retain_graph=True)
        
        if self.grad_clip: clip_grad_norm_(self.model.parameters(), 0.1)
        self.hept_opt.step()
        if self.lr_s: self.lr_s.step()
        

        with torch.no_grad():
            scores, labels = self.criterion.cluster(pos_embeds.cpu())
            targets = torch.where(pos_batch.hit_truth==3, 1, 0).cpu()
            
            auc = roc_auc_score(targets, scores[-1])
            loss_dict['sigbkg_auc'] = auc
                
            
        del pos_batch
        return loss_dict

    def run_one_epoch(self, pos_loader, epoch, phase):
        
        loader_len = len(pos_loader)
        run_one_batch = self.train_one_batch if phase == 'train' else self.eval_one_batch
        phase = 'test ' if phase == 'test' else phase  # align tqdm desc bar

        all_loss_dict = {}
        all_sample_idxs = []
        
        pbar = tqdm(pos_loader, total=loader_len)

        for idx, pos_batch in enumerate(pbar):
            
            if idx == 0:
                loss_dict = run_one_batch(pos_batch, new_epoch=True)
            else:
                loss_dict = run_one_batch(pos_batch)
            
            sample_idxs = pos_batch.sample_idx.cpu()
            
            del pos_batch
            
            torch.cuda.empty_cache()
            
            if epoch > -1:
                desc = log_epoch(epoch, phase, loss_dict, None, None, True, sample_idxs, reg=True)
                for k, v in loss_dict.items():
                    all_loss_dict[k] = all_loss_dict.get(k, 0) + v
                    
                all_sample_idxs.extend(list(sample_idxs))
    
                if idx == len(pbar)-1:
                #if len(all_clf_logits) > 10000:    
                #if idx == 5:
                    '''
                    if phase == 'train': # have to reset
                        
                        all_clf_logits, all_clf_labels, all_sample_idxs = [], [], []
                        for j, data in enumerate(data_loader):
                            loss_dict, clf_logits = self.eval_one_batch(data)
                            y = torch.cat([event.y for event in data]).cpu()
                        
                            all_clf_logits.extend(list(clf_logits)), all_clf_labels.extend(list(y)), all_sample_idxs.extend(list(sample_idxs))
                    '''
     
                    for k, v in all_loss_dict.items():
                        all_loss_dict[k] = v / loader_len
                    desc = log_epoch(epoch, phase, all_loss_dict, None, None, False, all_sample_idxs, writer=self.writer, reg=True)
                    break
            
            pbar.set_description(desc)
        
        
        return loss_dict['total_loss']

    def train(self):
        print(self.log_dir)
        start_epoch = 0
        if self.config['optimizer']['resume']:
            start_epoch = load_checkpoint(self.model, self.optimizer, self.log_path, self.device)

        best_val_loss = np.inf
        
        best_epoch = 0
        for epoch in range(start_epoch, self.config['optimizer']['epochs'] + 1):
            
            self.run_one_epoch(self.data_loaders['train'][0], epoch, 'train')
            
            if epoch == -1:
                continue
            
            valid_loss = self.run_one_epoch(self.data_loaders['valid'][0], epoch, 'valid')
            
            #self.lr_s.step(loss)
            if epoch % self.config['eval']['test_interval'] == 0:
                test_loss = self.run_one_epoch(self.data_loaders['test'][0], epoch, 'test')
                if valid_loss < best_val_loss:
                    save_checkpoint(self.model, self.hept_opt, self.log_path, epoch, criterion=self.criterion, name='model')
                    #save_checkpoint(self.decoder, self.dec_opt, self.log_path, epoch, name='decoder')
                    best_val_loss, best_test_loss, best_epoch = valid_loss, test_loss, epoch

            self.writer.add_scalar('best/best_epoch', best_epoch, epoch)
            self.writer.add_scalar('best/best_val_loss', best_val_loss, epoch)
            self.writer.add_scalar('best/best_test_loss', best_test_loss, epoch)
            
            print('-' * 50)
        
        
        print('Evaluating Performance')

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train Tau3MuGNNs')
    parser.add_argument('--setting', type=str, help='experiment settings', default='GNN_half_dR_1')
    parser.add_argument('--cut', type=str, help='cut id', default=None)
    parser.add_argument('--cuda', type=int, help='cuda device id, -1 for cpu', default=3)
    parser.add_argument('--comment', type=str, help='comment for log')
    parser.add_argument('--search', help='Use directory SearchConfigs instead of configs')
    
    args = parser.parse_args()
    setting = args.setting
    cuda_id = args.cuda
    cut_id = args.cut
    comment = args.comment
    search = args.search
    print(f'[INFO] Running {setting} on cuda {cuda_id} with cut {cut_id}')

    torch.set_num_threads(5)
    set_seed(42)
    time = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    
    config = yaml.safe_load(Path(f'./configs/{setting}.yml').open('r'))
    config = add_cuts_to_config(config, cut_id)
    path_to_config = Path(f'./configs/{setting}.yml')

    print(comment)
    device = torch.device(f'cuda:{cuda_id}' if cuda_id >= 0 else 'cpu')

    log_cut_name = '' if cut_id is None else f'-{cut_id}'
    
    if comment:
        log_name = f'{time}-{setting}{log_cut_name}_{comment}' if not config['optimizer']['resume'] else config['optimizer']['resume']
    else:
        log_name = f'{time}-{setting}{log_cut_name}' if not config['optimizer']['resume'] else config['optimizer']['resume']

    log_path = Path(config['data']['log_dir']) / log_name
    log_path.mkdir(parents=True, exist_ok=True)
    shutil.copy(f'{path_to_config}', log_path / 'config.yml')

    Tau3MuGNNs(config, device, log_path, setting).train()
    
if __name__ == '__main__':
    import os
    os.chdir('./src')
    main()
