import shutil
import torch
import torch.nn as nn
import yaml
from datetime import datetime
from tqdm import tqdm
from pathlib import Path

from utils import Criterion, Writer, log_epoch, load_checkpoint, save_checkpoint, set_seed, get_data_loaders, add_cuts_to_config
from models.get_model import get_model
#from model_efficiency import main as model_eff
#from model_efficiency import eval_one_batch, run_one_epoch, generate_roc
from torch.optim.lr_scheduler import ReduceLROnPlateau

        
class Tau3MuGNNs:

    def __init__(self, config, device, log_path, setting):
        self.config = config
        self.device = device
        self.log_path = log_path
        self.writer = Writer(log_path)
        self.log_dir=self.writer.log_dir
        
        endcap = config['model_kwargs']['endcap']
        
        lr_s_kwargs = config['lr_scheduler_kwargs']
        
        self.data_loaders, x_dim, dataset = get_data_loaders(setting, config['data'], config['optimizer']['batch_size'], endcap=endcap)
        
        self.model = get_model(config['model_kwargs'],dataset)   
        #self.init_model(self.model)
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config['optimizer']['lr'])
        #self.lr_s = ReduceLROnPlateau(self.optimizer, **lr_s_kwargs)
        self.criterion = Criterion(config['optimizer'])
        self.node_clf = config['data'].get('node_clf', False)
      
    
    def init_model(self, model):
        for n,l in model.named_modules():
            if isinstance(l, nn.Linear): 
                torch.nn.init.xavier_uniform_(l.weight)
        
    @torch.no_grad()
    def eval_one_batch(self, batch):
        self.model.eval()
        logits = []
        
        batch_size = len(batch)
        if batch_size == 1:
            data = batch[0]
            data.to(self.device)
            event_clf_logits = self.model(data)
            y = data.y
            
            loss, loss_dict = self.criterion(event_clf_logits.sigmoid(), y)
            logits.append(event_clf_logits.data.cpu())
        else:
            assert batch_size > 0
            
            total_loss = 0
            
            for data in batch:
                data.to(self.device)
                event_clf_logits = self.model(data)
                y = data.y
                
                loss, _t = self.criterion(event_clf_logits.sigmoid(), y)
                total_loss += loss
                logits.append(event_clf_logits.data.cpu())
            
            total_loss /= batch_size
            loss_dict = {'total':total_loss.item(), 'focal':total_loss.item()}
            
        logits = torch.cat(logits)
        return loss_dict, logits
        
    def train_one_batch(self, batch):
        self.model.train()
        logits = []
        
        batch_size = len(batch)
        if batch_size == 1:
            data = batch[0]
            data.to(self.device)
            event_clf_logits = self.model(data)
            y = data.y
            
            loss, loss_dict = self.criterion(event_clf_logits.sigmoid(), y)
            logits.append(event_clf_logits.data.cpu())
            
        else:
            assert batch_size > 0
            
            total_loss = 0
            
            for data in batch:
                data.to(self.device)
                event_clf_logits = self.model(data)
                y = data.y
                
                loss, _ = self.criterion(event_clf_logits.sigmoid(), y)
                total_loss += loss
                logits.append(event_clf_logits.data.cpu())
            
            total_loss /= batch_size
            loss_dict = {'total':total_loss.item(), 'focal':total_loss.item()}
            loss = total_loss
        
        logits = torch.cat(logits)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss_dict, logits

    def run_one_epoch(self, data_loader, epoch, phase):
        loader_len = len(data_loader)
        run_one_batch = self.train_one_batch if phase == 'train' else self.eval_one_batch
        phase = 'test ' if phase == 'test' else phase  # align tqdm desc bar

        all_loss_dict, all_clf_logits, all_clf_labels, all_sample_idxs = {}, [], [], []
        pbar = tqdm(data_loader, total=loader_len)

        for idx, batch in enumerate(pbar):
        
            loss_dict, clf_logits = run_one_batch(batch)
            y = torch.cat([data.y.cpu() for data in batch])
            sample_idxs = torch.cat([data.sample_idx.cpu() for data in batch])
            
            clf_logits = clf_logits.cpu()
            
            desc = log_epoch(epoch, phase, loss_dict, clf_logits, y, True, sample_idxs)
            for k, v in loss_dict.items():
                all_loss_dict[k] = all_loss_dict.get(k, 0) + v
                
            all_clf_logits.extend(list(clf_logits)), all_clf_labels.extend(list(y)), all_sample_idxs.extend(list(sample_idxs))

            #if idx == min([10, len(pbar)-1]):
            if len(all_clf_logits) > 10000:    
                '''
                if phase == 'train': # have to reset
                    
                    all_clf_logits, all_clf_labels, all_sample_idxs = [], [], []
                    for j, data in enumerate(data_loader):
                        loss_dict, clf_logits = self.eval_one_batch(data)
                        y = torch.cat([event.y for event in data]).cpu()
                    
                        all_clf_logits.extend(list(clf_logits)), all_clf_labels.extend(list(y)), all_sample_idxs.extend(list(sample_idxs))
                '''
                all_clf_logits, all_clf_labels = torch.cat(all_clf_logits), torch.cat(all_clf_labels)
                for k, v in all_loss_dict.items():
                    all_loss_dict[k] = v / loader_len
                desc, auroc, recall, avg_loss = log_epoch(epoch, phase, all_loss_dict, all_clf_logits, all_clf_labels, False, all_sample_idxs, writer=self.writer)
                
                signal_mask = (all_clf_labels==1)
                bkg_mask = (all_clf_labels==0)
                
                self.writer.add_histogram(f'{phase}/Signal_Predictions', all_clf_logits[signal_mask].sigmoid(),epoch)
                self.writer.add_histogram(f'{phase}/Signal_Logits', all_clf_logits[signal_mask],epoch)
                self.writer.add_histogram(f'{phase}/Bkg_Predictions', all_clf_logits[bkg_mask].sigmoid(),epoch)
                self.writer.add_histogram(f'{phase}/Bkg_Logits', all_clf_logits[bkg_mask],epoch)
                break
            
            pbar.set_description(desc)

        return avg_loss, auroc, recall

    def train(self):
        print(self.log_dir)
        start_epoch = 0
        if self.config['optimizer']['resume']:
            start_epoch = load_checkpoint(self.model, self.optimizer, self.log_path, self.device)

        best_val_auc = 0
        best_test_auc = 0
        best_test_auc = 0
        
        best_epoch = 0
        for epoch in range(start_epoch, self.config['optimizer']['epochs'] + 1):
            self.run_one_epoch(self.data_loaders['train'], epoch, 'train')
            loss, valid_auc = self.run_one_epoch(self.data_loaders['valid'], epoch, 'valid')[:2]
            
            #self.lr_s.step(loss)
            if epoch % self.config['eval']['test_interval'] == 0:
                test_auc = self.run_one_epoch(self.data_loaders['test'], epoch, 'test')[1]
                if valid_auc > best_val_auc:
                    save_checkpoint(self.model, self.optimizer, self.log_path, epoch)
                    best_val_auc, best_test_auc, best_epoch = valid_auc, test_auc, epoch

            self.writer.add_scalar('best/best_epoch', best_epoch, epoch)
            self.writer.add_scalar('best/best_val_auc', best_val_auc, epoch)
            self.writer.add_scalar('best/best_test_auc', best_test_auc, epoch)
            
            print('-' * 50)
        
        save_checkpoint(self.model, self.optimizer, self.log_path, epoch)
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
