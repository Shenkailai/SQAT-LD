
import torch
import torch.nn.functional as F
from torch import nn


class AttDot(torch.nn.Module):
    '''
    AttDot: Dot attention that can be used by the Alignment module.
    '''       
    def __init__(self):
        super().__init__()
    def forward(self, query, y):
        att = torch.bmm(query, y.transpose(2,1))
        sim = att.max(2)[0].unsqueeze(1)
        return att, sim
    
class AttCosine(torch.nn.Module):
    '''
    AttCosine: Cosine attention that can be used by the Alignment module.
    '''          
    def __init__(self):
        super().__init__()
        self.pdist = nn.CosineSimilarity(dim=3)
    def forward(self, query, y):
        att = self.pdist(query.unsqueeze(2), y.unsqueeze(1))
        sim = att.max(2)[0].unsqueeze(1)
        return att, sim    
    
class AttDistance(torch.nn.Module):
    '''
    AttDistance: Distance attention that can be used by the Alignment module.
    '''        
    def __init__(self, dist_norm=1, weight_norm=1):
        super().__init__()
        self.dist_norm = dist_norm
        self.weight_norm = weight_norm
    def forward(self, query, y):
        att = (query.unsqueeze(1)-y.unsqueeze(2)).abs().pow(self.dist_norm)
        att = att.mean(dim=3).pow(self.weight_norm)
        att = - att.transpose(2,1)
        sim = att.max(2)[0].unsqueeze(1)
        return att, sim
    
class AttBahdanau(torch.nn.Module):
    '''
    AttBahdanau: Attention according to Bahdanau that can be used by the 
    Alignment module.
    ''' 
    def __init__(self, q_dim, y_dim, att_dim=128):
        super().__init__()
        self.q_dim = q_dim
        self.y_dim = y_dim
        self.att_dim = att_dim
        self.Wq = nn.Linear(self.q_dim, self.att_dim)
        self.Wy = nn.Linear(self.y_dim, self.att_dim)
        self.v = nn.Linear(self.att_dim, 1)
    def forward(self, query, y):
        att = torch.tanh( self.Wq(query).unsqueeze(1) + self.Wy(y).unsqueeze(2) )
        att = self.v(att).squeeze(3).transpose(2,1)
        sim = att.max(2)[0].unsqueeze(1)
        return att, sim

class AttLuong(torch.nn.Module):
    '''
    AttLuong: Attention according to Luong that can be used by the 
    Alignment module.
    '''     
    def __init__(self, q_dim, y_dim):
        super().__init__()
        self.q_dim = q_dim
        self.y_dim = y_dim
        self.W = nn.Linear(self.y_dim, self.q_dim)
    def forward(self, query, y):
        att = torch.bmm(query, self.W(y).transpose(2,1))
        sim = att.max(2)[0].unsqueeze(1)
        return att, sim

class ApplyHardAttention(torch.nn.Module):
    '''
    ApplyHardAttention: Apply hard attention for the purpose of time-alignment.
    '''       
    def __init__(self):
        super().__init__()
    def forward(self, y, att):        
        self.idx = att.argmax(2)
        y = y[torch.arange(y.shape[0]).unsqueeze(-1), self.idx]        
        return y    
    
class ApplySoftAttention(torch.nn.Module):
    '''
    ApplySoftAttention: Apply soft attention for the purpose of time-alignment.
    '''        
    def __init__(self):
        super().__init__()
    def forward(self, y, att):        
        y = torch.bmm(att, y)       
        return y     

class Alignment(torch.nn.Module):
    '''
    Alignment: Alignment module for the model. It 
    supports five different alignment mechanisms.
    '''       
    def __init__(self,
                 att_method,
                 apply_att_method,
                 q_dim=None,
                 y_dim=None,
                 ):
        super().__init__()
        
        
        # Attention method --------------------------------------------------------
        if att_method=='bahd':
            self.att = AttBahdanau(
                    q_dim=q_dim,
                    y_dim=y_dim) 
            
        elif att_method=='luong':
            self.att = AttLuong(
                    q_dim=q_dim, 
                    y_dim=y_dim) 
            
        elif att_method=='dot':
            self.att = AttDot()
            
        elif att_method=='cosine':
            self.att = AttCosine()            

        elif att_method=='distance':
            self.att = AttDistance()
            
        elif (att_method=='none') or (att_method is None):
            self.att = None
        else:
            raise NotImplementedError  
        
        # Apply method ----------------------------------------------------------
        if apply_att_method=='soft':
            self.apply_att = ApplySoftAttention() 
        elif apply_att_method=='hard':
            self.apply_att = ApplyHardAttention() 
        else:
            raise NotImplementedError    
        
        
    def forward(self, query, y):        
        if self.att is not None:
            att_score, sim = self.att(query, y)     
            att_score = F.softmax(att_score, dim=2)
            y = self.apply_att(y, att_score) 
        return y        
