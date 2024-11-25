import torch
import torch.nn as nn
import torch.nn.functional as F
class ContrastiveLoss(nn.Module):
    def __init__(self, device='cuda', temperature=0.5):
        super().__init__()

        self.temperature = 0.1
        self.base_temperature = 2
        
    def forward(self, e1,e2,labels):
        e1 = e1/e1.norm(dim=-1, keepdim=True)
        e2 = e2/e2.norm(dim=-1, keepdim=True)
        #print("e1,e2",e1.shape,e2.shape)
        # emb_neg =emb_neg/emb_neg.norm(dim=-1, keepdim=True)
        logits= torch.einsum('nc,nkc->nk', [e1, e2])
        # l_neg = torch.einsum('nc,nkc->nk', [emb_i, emb_neg])
        # # logits: Nx(1+K)
        # logits = torch.cat([l_pos, l_neg], dim=1)
        # apply temperature
        logits /= 0.07
        #labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        criterion = nn.CrossEntropyLoss().cuda()

        loss = criterion(logits+1e-8, labels)
       

        return loss
    



class ContrastiveLoss_ppc(nn.Module):
    def __init__(self, device='cuda', temperature=0.5):
        super().__init__()
        
    def forward(self, emb_i, emb_j,emb_neg):
        emb_i = emb_i/emb_i.norm(dim=-1, keepdim=True)
        emb_j= emb_j/emb_j.norm(dim=-1, keepdim=True)
        emb_neg =emb_neg/emb_neg.norm(dim=-1, keepdim=True)
        l_pos = torch.einsum('nc,nc->n', [emb_i, emb_j]).unsqueeze(-1)
        l_neg = torch.einsum('nc,nkc->nk', [emb_i, emb_neg])
        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)
        # apply temperature
        logits /= 0.07
        #print("logits",logits)
        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        criterion = nn.CrossEntropyLoss().cuda()

        loss = criterion(logits, labels)
        return loss



     