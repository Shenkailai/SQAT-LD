import torch
import torch.nn as nn

from models.align import Alignment
from .ssast_models import SSASTModel
from einops import rearrange


class TABlock(nn.Module):
    def __init__(self, dim, drop=0.1):
        super().__init__()
        self.c_q = nn.Linear(dim, dim)
        self.c_k = nn.Linear(dim, dim)
        self.c_v = nn.Linear(dim, dim)
        self.norm_fact = dim ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.proj_drop = nn.Dropout(drop)

    def forward(self, x):
        _x = x
        B, C, N = x.shape
        q = self.c_q(x)
        k = self.c_k(x)
        v = self.c_v(x)

        attn = q @ k.transpose(-2, -1) * self.norm_fact
        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, C, N)
        x = self.proj_drop(x)
        x = x + _x
        return x


class SQAT_LD(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()

        drop = 0.1
        self.scale = args['scale']
        # input_fdim // fshape
        self.f_dim = args['mel_bins'] // args['fshape'] 
        # input_tdim // tshape
        self.t_dim = args['target_length'] // args['tshape']
        self.embed_dim = args['embed_dim']
        self.window_size = args['window_size']
        self.dim_mlp = args['dim_mlp']
        self.num_heads = args['num_heads']
        self.depths = args['depths']
        self.num_tab = args['num_tab']
        self.att_method = args['att_method']
        self.apply_att_method = args['apply_att_method']
        self.args = args
        
         # define judge embedding
        self.num_judges = args["num_judges"]
        self.judge_embedding = nn.Embedding(num_embeddings = self.num_judges, embedding_dim = args['embed_dim'])
        
        self.ast = SSASTModel(label_dim=1, fshape=args['fshape'], tshape=args['tshape'], fstride=args['fstride'], tstride=args['tstride'],
                              input_fdim=args['mel_bins'], input_tdim=args['target_length'], model_size=args['model_size'],
                              pretrain_stage=False, load_pretrained_mdl_path=args['load_pretrained_mdl_path'])
        self.tablock1 = nn.ModuleList()
        for i in range(self.num_tab):
            tab = TABlock(self.f_dim * self.t_dim)
            self.tablock1.append(tab)
        self.conv1 = nn.Conv2d(self.embed_dim * 2, self.embed_dim, 1, 1, 0)
        self.align = Alignment(
             self.att_method, 
             self.apply_att_method,
             q_dim=self.embed_dim,
             y_dim=self.embed_dim,
            )
        
        
        self.fc_score = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(self.embed_dim, 2),
            nn.ReLU()
        )
        self.fc_weight = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(self.embed_dim, 2),
            nn.Sigmoid()
        )
        
        self.mlp_head = nn.Sequential(nn.LayerNorm(
            self.embed_dim), nn.Linear(self.embed_dim, 1))
    
    def forward(self, x, judge_id):
        x = self.ast(x)  # b 512 768
        judge_feat = self.judge_embedding(judge_id) # (batch, emb_dim)
        judge_feat = torch.stack([judge_feat for i in range(self.f_dim * self.t_dim)], dim = 1) #(batch, F, D)
       
        
        judge_feat = self.align(x, judge_feat)
        x = torch.concat((x, judge_feat),dim=2)
        
        # stage 1
        x = rearrange(x, 'b (h w) c -> b c (h w)',
                      h=self.f_dim, w=self.t_dim)  # 1 768 512
        for tab in self.tablock1:
            x = tab(x)
        x = rearrange(x, 'b c (h w) -> b c h w', h=self.f_dim,
                      w=self.t_dim)  # 1 768 28 28
        x = self.conv1(x)
        # x = self.swintransformer1(x)

        x = rearrange(x, 'b c h w -> b (h w) c', h=self.f_dim, w=self.t_dim)

        score = torch.tensor([]).cuda()
        for i in range(x.shape[0]):
            f = self.fc_score(x[i])
            w = self.fc_weight(x[i])
            _s = torch.sum(f * w) / torch.sum(w)
            score = torch.cat((score, _s.unsqueeze(0)), 0)
        score = score.unsqueeze(1)
        
        judge_score = torch.tensor([]).cuda()
        for i in range(x.shape[0]):
            jf = self.fc_score(x[i])
            jw = self.fc_weight(x[i])
            j_s = torch.sum(jf * jw) / torch.sum(jw)
            judge_score = torch.cat((judge_score, j_s.unsqueeze(0)), 0)
        judge_score = judge_score.unsqueeze(1)
        # score = (score + clip_lable)/2
        return score, judge_score


if __name__ == '__main__':
    input_tdim = 1024
    ast_mdl = SQAT_LD(input_tdim=input_tdim, fstride=128,
                    fshape=128, tstride=2, tshape=2).cuda()
    # input a batch of 10 spectrogram, each with 100 time frames and 128 frequency bins
    test_input = torch.rand([1, input_tdim, 128]).cuda()
    test_output = ast_mdl(test_input)
    # output should be in shape [10, 527], i.e., 10 samples, each with prediction of 527 classes.
    print(test_output.shape)

