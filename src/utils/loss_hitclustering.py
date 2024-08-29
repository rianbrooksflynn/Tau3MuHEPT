import torch
import torch.nn.functional as F
from torch.nn import CosineSimilarity, MSELoss
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score

class ClusterLoss(torch.nn.Module):
    
    def __init__(self, optimizer_config):
        super(ClusterLoss, self).__init__()
        self.temp =  optimizer_config['temp']
        self.sim = CosineSimilarity(dim=1, eps=1e-2)
        
        self.mse = MSELoss()
        self.sig_centroid = 0
        self.bkg_centroid = 0
        self.tracked_sig_batches = []
        self.tracked_bkg_batches = []
        
    def forward(self, embeds, truth, train=True):

        sig = embeds[ truth==3 ]
        neg = embeds[ truth!=3 ]
        
        if len(sig) == 0:
            raise
            
        if train:
            self.sig_centroid = sig.mean(dim=0)
            self.bkg_centroid = neg.mean(dim=0)
        
        #sig_loss = ((sig - self.sig_centroid)**2).mean()
        sig_loss = (torch.exp(-self.sim(self.sig_centroid.repeat(sig.shape[0],1), sig)/self.temp)-torch.exp(torch.tensor(-1/self.temp))).mean() 
        neg_loss = (torch.exp(-self.sim(self.bkg_centroid.repeat(neg.shape[0],1), neg)/self.temp)-torch.exp(torch.tensor(-1/self.temp))).mean()

        centroid_loss = torch.exp(CosineSimilarity(dim=0,eps=1e-2)(self.sig_centroid, self.bkg_centroid)/self.temp)-torch.exp(torch.tensor(-1/self.temp))
        
        #scale_factor = (neg_loss.clone().detach())/(sig_loss.clone().detach())
        
        #centroid_loss = torch.exp(-self.temp*self.mse(self.sig_centroid, self.bkg_centroid))
        '''
        if centroid_loss.clone().detach() < sig_loss.clone().detach()+neg_loss.clone().detach():
            loss =  sig_loss + neg_loss
        else:    
            loss = centroid_loss
         '''
         
        loss = centroid_loss + sig_loss + neg_loss
        loss_dict = {'signal_hits':sig_loss.clone().detach().cpu(), 'pileup_hits':neg_loss.clone().detach().cpu(), 'centroid':centroid_loss.clone().detach().cpu()}

        return loss, loss_dict
    
    def cluster(self, embeds):
        
        centroids = torch.stack( [self.bkg_centroid, self.sig_centroid] )
        labels = []
        dists = []
        with torch.no_grad():
  
            for i, centroid in enumerate(centroids):
                dists.append(self.sim(embeds.cpu(), centroid.cpu()))
            
            dists = torch.stack(dists)
            _, labels = torch.max(dists,dim=0)
            scores = F.softmax(dists,dim=0)[1]
            
            return scores, labels

class MultiClusterLoss(torch.nn.Module):
    
    def __init__(self, optimizer_config, in_dim, num_clusters=4):
        super(MultiClusterLoss, self).__init__()
        self.temp =  optimizer_config['temp']
        self.sim = CosineSimilarity(dim=1, eps=1e-2)
        self.mse = MSELoss()
        
        self.centroids = [torch.rand(in_dim) for i in range(num_clusters)]
        
        
    def forward(self, embeds, truth, train=True):
        
        sorted_embeds = {int(i): embeds[truth==i] for i in torch.unique(truth)}
        if train:
            for i in sorted_embeds.keys():
                self.centroids[i] = sorted_embeds[i].mean(dim=0)
        
        embed_loss = 0
        for i in sorted_embeds.keys():
            embed_loss += (torch.exp(-self.sim(self.centroids[i].repeat(sorted_embeds[i].shape[0],1), sorted_embeds[i])/self.temp)-torch.exp(torch.tensor(-1/self.temp))).mean() 
        
        centroid_loss = 0
        for i in range(len(self.centroids)):
            for j in range(i+1,len(self.centroids)):
                
                centroid_loss += torch.exp(CosineSimilarity(dim=0,eps=1e-2)(self.centroids[i], self.centroids[j])/self.temp)-torch.exp(torch.tensor(-1/self.temp))
        
        loss = centroid_loss + embed_loss
        loss_dict = {'total_loss': loss.clone().detach().cpu(),'embed_loss':embed_loss.clone().detach().cpu(), 'centroid_loss':centroid_loss.clone().detach().cpu()}

        return loss, loss_dict
    
    def cluster(self, embeds):
        
        centroids = torch.stack( self.centroids )
        labels = []
        dists = []
        with torch.no_grad():
  
            for i, centroid in enumerate(centroids):
                dists.append(self.sim(embeds.cpu(), centroid.cpu()))
            
            dists = torch.stack(dists)
            _, labels = torch.max(dists,dim=0)
            scores = F.softmax(dists,dim=0)
            
            return scores, labels


class ClusterLossUnsup(torch.nn.Module):
    
    def __init__(self, optimizer_config):
        super(ClusterLossUnsup, self).__init__()
        self.temp =  optimizer_config['temp']
        self.sim = CosineSimilarity(dim=1, eps=1e-2)
        
        self.mse = MSELoss()

        
    def forward(self, embeds, comparator, train=True):

        if train:
            self.centroid = embeds.mean(dim=0)
        

        #sig_loss = ((sig - self.sig_centroid)**2).mean()
        loss = (torch.exp(-self.sim(comparator.repeat(embeds.shape[0],1), embeds)/self.temp)-torch.exp(torch.tensor(-1/self.temp))).mean() 

        loss_dict = {'contrastive_loss': loss}

        return loss, loss_dict
    
    def clusterROC(self, embeds, comparator, truth):
        
        logits = (-self.sim(comparator.repeat(embeds.shape[0],1), embeds) + 1)/2
        
        labels = torch.where(truth.cpu()==3, 1, 0)
        
        
        fpr, tpr, _ = roc_curve(labels, logits.cpu())
        auc = roc_auc_score(labels, logits.cpu())
        
        return fpr, tpr, auc
        
        
        
        