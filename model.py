import torch
import torch.nn as nn
import numpy as np
from HeteroHG_Batch import Encoder_HeteroHG_transformer_batch
from config import args


device = torch.device('cuda:{}'.format(args.gpu))
class Embedding(nn.Module):
    def __init__(self, vocab_size, max_len, d_model):
        super(Embedding, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)  #

    def forward(self, x):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long).to(device)
        pos = pos.unsqueeze(0).expand_as(x).to(device)

        embedding = self.tok_embed(x) + self.pos_embed(pos)
        return self.norm(embedding)



class Predictor(nn.Module):
    def __init__(self, eventlog, d_model,  f):
        super(Predictor, self).__init__()

        self.encoder = Encoder_HeteroHG_transformer_batch(
            eventlog=eventlog,
            d_model=d_model,
            f=f
        )

        self.activity_vocab_size = np.load(
            "data/" + eventlog + "/" + eventlog + '_' + 'activity' + '_' + str(f) + "_info.npy")
        self.feat_dim =d_model


        self.fc = nn.Linear(self.feat_dim, self.feat_dim)
        self.active = nn.Tanh()
        self.projection = nn.Linear(self.feat_dim, self.activity_vocab_size, bias=False)

        self.classifer = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.feat_dim, self.activity_vocab_size, bias=False)
        )


        torch.nn.init.xavier_uniform_(self.fc.weight)
        torch.nn.init.xavier_uniform_(self.projection.weight)

    def forward(self, att_str, att):

        global_feat, remian_logits = self.encoder(att_str, att)

        feat = self.active(self.fc(global_feat))  # [B, feat_dim]
        dec_logits = self.projection(feat)  # [B, activity_vocab_size]


        return dec_logits, [], remian_logits

