import yaml
from pathlib import Path
from utils import get_data_loaders, load_checkpoint, log_epoch, Criterion, add_cuts_to_config
import torch
from models import Model
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from torch_geometric.nn import DataParallel

def bin_acc(preds, labels, threshold=.5):
    preds = torch.where(preds<threshold, 0, 1)
    
    acc = ((preds==labels).sum())/len(preds)
    
    return acc
        
@torch.no_grad()
def eval_one_batch(data, model, criterion, node_clf=False):
    model.eval()
    device = torch.device('cuda:0')
    y = torch.cat([event.y for event in data]).to(device)
    event_clf_logits = model(data)
    event_loss, loss_dict = criterion(event_clf_logits.sigmoid(), y)
    
    if node_clf:
        node_clf_logits = model(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch, ptr=data.ptr, node_clf=True)
        node_loss, _ = criterion(node_clf_logits.sigmoid(), data.node_label)
        
        loss = event_loss+ node_clf*node_loss
        loss_dict['node_focal'] = node_loss
        node_acc = bin_acc(node_clf_logits.sigmoid(), data.node_label)
        loss_dict['node_acc'] = node_acc
    
    return loss_dict, event_clf_logits.data.cpu(), y


def run_one_epoch(data_loader, epoch, phase, device, model, criterion):
    loader_len = len(data_loader)
    run_one_batch = eval_one_batch
    phase = 'test ' if phase == 'test' else phase  # align tqdm desc bar

    all_loss_dict, all_clf_logits, all_clf_labels, all_sample_idx, all_endcap = {}, [], [], [], []
    all_labels = []
    
    pbar = tqdm(data_loader, total=loader_len)
    for idx, data in enumerate(pbar):
        loss_dict, clf_logits, y = run_one_batch(data, model, criterion)
        
        y = torch.cat([event.y for event in data]).to(device)
        sample_idx = torch.cat([event.sample_idx for event in data]).to(device)
        endcap = torch.cat([event.endcap for event in data]).to(device)
        
        desc = log_epoch(epoch, phase, loss_dict, clf_logits, y.data.cpu(), True, sample_idx)
        for k, v in loss_dict.items():
            all_loss_dict[k] = all_loss_dict.get(k, 0) + v
        
        all_clf_logits.extend(list(clf_logits.sigmoid())), all_clf_labels.append(y.data.cpu())
        all_sample_idx.append(sample_idx.data.cpu()), all_endcap.append(endcap.data.cpu())
        all_labels.extend(list(y))
        
        if idx == loader_len - 1:
            all_clf_logits, all_clf_labels = torch.cat(all_clf_logits), torch.cat(all_clf_labels)
            all_sample_idx, all_endcap = torch.cat(all_sample_idx), torch.cat(all_endcap)
            phase_idx = [phase.strip()] * len(all_sample_idx)
            for k, v in all_loss_dict.items():
                all_loss_dict[k] = v / loader_len
            desc, auroc, recall, avg_loss = log_epoch(epoch, phase, all_loss_dict, all_clf_logits, all_clf_labels, False, all_sample_idx, None)
        pbar.set_description(desc)
    
    print(all_sample_idx)
    all_sample_idx = list(all_sample_idx)
    all_sample_idx = [int(i) for i in all_sample_idx]
    return avg_loss, auroc, recall, all_clf_logits, all_sample_idx, all_endcap, phase_idx, all_labels

def get_idx_for_interested_fpr(fpr, interested_fpr):
    res = []
    for i in range(1, fpr.shape[0]):
        if fpr[i] > interested_fpr:
            res.append(i-1)
            break
    return res

def generate_roc(labels, predictions, setting, show=False, save=False):
    

    fpr, tpr, thresholds = metrics.roc_curve(labels, predictions)
    roc_auc = metrics.auc(fpr, tpr)
    
    threshes = {}
    
    R_LHC = 2760*11.246
    
    fpr_rate = fpr*R_LHC
    
    lw = 2
    plt.clf()
    plt.semilogx(fpr_rate, tpr, color='darkorange',
            lw=lw, label='ROC (area = %0.4f)' % roc_auc)
    plt.xlim((1, 12000))
    plt.ylim([0.0, 1.05])
    plt.xlabel('Trig Rate (kHz)')
    plt.ylabel('Trigger Acceptance')
    plt.title(f'{setting} ROC Curve')
    plt.grid()
    
    print('AUROC: ', roc_auc)
    colors = ['purple', 'blue', 'green', 'orange', 'red']
    for i, rate in enumerate([30.00, 100.00, 200.00, 1000.00, 10000.00]):
    
        max_fpr = rate/R_LHC
        index = get_idx_for_interested_fpr(fpr, max_fpr)
        interested_tpr = float(tpr[index])
        thresh = float(thresholds[index])
        print('Threshold at {0}kHz:'.format(int(rate)), thresh)
        threshes[str(rate)+'kHz'] = thresh
        plt.axvline(x = max_fpr*R_LHC, label="Trig Rate: {0:.2f}kHz\nTrig Acc: {1:.4f}".format(max_fpr*R_LHC,interested_tpr), linestyle='dashed', c=colors[i])
    
    plt.legend(loc="upper left")
    if show:
        plt.show()
    
    if save:
        plt.savefig(f'{save}/roc.png')
    
def main(log_name, setting):

    cuda_id = 0
    
    config = yaml.safe_load(Path(f'{log_name}/config.yml').open('r'))
    endcap = config['model']['endcap']
    #config = yaml.safe_load(Path(f'./configs/{setting}.yml').open('r'))
    node_clf = config['data'].get('node_clf', False)
    
    device = torch.device(f'cuda:{cuda_id}' if cuda_id >= 0 else 'cpu')
    data_loaders, x_dim, edge_attr_dim, dataset = get_data_loaders(setting, config['data'], 64, endcap=endcap)
    
    model = Model(x_dim, edge_attr_dim, config['data']['virtual_node'], config['model']).to(device)
    model = DataParallel(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['optimizer']['lr'])
    
    load_checkpoint(model, optimizer, Path(log_name), device)
    criterion = Criterion(config['optimizer'])
    
    clf_probs, all_sample_idx, all_endcap, all_phase_idx, all_labels = [], [], [], [], []
    sample_dict = {}
    
    for phase in ['valid', 'test']:
        avg_loss, auroc, recall, clf_logits, sample_idx, endcap, phase_idx, labels = run_one_epoch(data_loaders[phase], 999, phase, device, model, criterion)
        clf_probs.extend(clf_logits)
        all_sample_idx.extend(sample_idx)
        
        all_endcap.append(endcap)
        all_labels.extend(labels)
        all_phase_idx.extend(phase_idx)
    
    labels = [int(x) for x in all_labels]
    predictions = [float(p) for p in clf_probs]
    
    #print(all_sample_idx)
    for i in range(len(all_sample_idx)):
        idx = all_sample_idx[i]
        
        if idx not in sample_dict.keys():
                sample_dict[idx] = [predictions[i], labels[i]]
        else:
            #print(idx)
            sample_dict[idx] = [max(predictions[i], sample_dict[idx][0]), max(labels[i], sample_dict[idx][1])]
    
    #print(sample_dict.keys())
    labels = []
    predictions = []
    
    for value in sample_dict.values():
        pred = value[0]
        label = value[1]
        
        predictions.append(pred)
        labels.append(label)
    
    df = pd.DataFrame.from_dict(sample_dict)
    df.to_pickle(log_name+'/valid_test_predictions.pkl')
    
    clf_probs, all_sample_idx, all_endcap, all_phase_idx, all_labels = [], [], [], [], []
    sample_dict = {}
    
    for phase in ['train']:
        avg_loss, auroc, recall, clf_logits, sample_idx, endcap, phase_idx, labels = run_one_epoch(data_loaders[phase], 999, phase, device, model, criterion)
        clf_probs.extend(clf_logits)
        all_sample_idx.extend(sample_idx)
        
        all_endcap.append(endcap)
        all_labels.extend(labels)
        all_phase_idx.extend(phase_idx)
    
    labels = [int(x) for x in all_labels]
    predictions = [float(p) for p in clf_probs]
    
    #print(all_sample_idx)
    for i in range(len(all_sample_idx)):
        idx = all_sample_idx[i]
        
        if idx not in sample_dict.keys():
                sample_dict[idx] = [predictions[i], labels[i]]
        else:
            #print(idx)
            sample_dict[idx] = [max(predictions[i], sample_dict[idx][0]), max(labels[i], sample_dict[idx][1])]
    
    #print(sample_dict.keys())
    labels = []
    predictions = []
    
    for value in sample_dict.values():
        pred = value[0]
        label = value[1]
        
        predictions.append(pred)
        labels.append(label)
    
    df = pd.DataFrame.from_dict(sample_dict)
    df.to_pickle(log_name+'/train_predictions.pkl')
    
    #generate_roc(labels, predictions, setting, save=log_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_name', help="Path to log directory'", type=str)
    parser.add_argument('--setting', help="GNN setting", type=str)
    
    args = parser.parse_args()
    
    log_name = args.log_name
    setting = args.setting
    
    main(log_name, setting)







