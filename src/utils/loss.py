import torch
import torch.nn.functional as F
from torch.nn import CosineSimilarity

class Criterion(torch.nn.Module):
    
    def __init__(self,optimizer_config):
        super(Criterion, self).__init__()
        self.alpha = optimizer_config['total_alpha']
        self.focal_loss = FocalLoss(optimizer_config)
        self.contrastive_loss = ContrastiveLoss(optimizer_config)
    
    def forward(self, embeds, logits):
        
        loss_dict = {}
        cl = self.contrastive_loss(embeds)
        
        pos_logits, neg_logits = logits
        pos_scores = F.sigmoid(pos_logits)
        neg_scores = F.sigmoid(neg_logits)
        
        pred_scores = torch.cat([pos_scores, neg_scores]).view(-1)
        targets = torch.cat([torch.ones(len(pos_logits)), torch.zeros(len(neg_logits))]).to(pred_scores.device)
        
        fl = self.focal_loss(pred_scores, targets)
        
        total = self.alpha*(cl)+(1-self.alpha)*(fl)
        
        loss_dict['focal'] = fl.cpu().item()
        loss_dict['contrastive'] = cl.cpu().item()
        loss_dict['total'] = total.cpu().item()
        return fl, total, loss_dict

class FocalLoss(torch.nn.Module):

    def __init__(self, optimizer_config):
        super(FocalLoss, self).__init__()
        self.alpha = optimizer_config['focal_alpha']
        self.gamma = optimizer_config['focal_gamma']

    def forward(self, scores, targets):

        inputs = torch.clamp(scores, min=1e-5, max=1-1e-5)
        
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
        p_t = inputs * targets + (1 - inputs) * (1 - targets)
        loss = bce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss
            
        loss = loss.mean()

        return loss

class ContrastiveLoss(torch.nn.Module):
    
    def __init__(self, optimizer_config):
        super(ContrastiveLoss, self).__init__()
        self.temp =  optimizer_config['temp']
        self.sim = CosineSimilarity(eps=1e-2)
        
    def forward(self, inputs):
        sig, aug, neg = inputs
        
        sig_aug = torch.exp(self.sim(sig,aug)/self.temp)
        sig_neg = torch.exp(self.sim(sig,neg)/self.temp)
        aug_neg = torch.exp(self.sim(aug,neg)/self.temp)
        
        loss = -torch.log((sig_aug/(sig_aug+sig_neg+aug_neg))+1e-2)
        loss = loss.mean()

        return loss
        
        
        
        
        
        
        