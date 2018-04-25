import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from net_utils import run_lstm, col_name_encode

class SelPredictor(nn.Module):
    def __init__(self, N_word, N_h, N_depth, max_tok_num, use_ca):
        super(SelPredictor, self).__init__()
        self.use_ca = use_ca
        self.max_tok_num = max_tok_num
        self.sel_lstm = nn.LSTM(input_size=N_word+N_h, hidden_size=N_h/2,
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)
        
        if use_ca:
            print "Using column attention on selection predicting"
            self.sel_att = nn.Linear(N_h, N_h)
            self.sel_agg_att = nn.Linear(N_h, N_h)
            self.sel_type_att = nn.Linear(N_h, N_h)
        else:
            print "Not using column attention on selection predicting"
            self.sel_att = nn.Linear(N_h, 1)
            self.sel_agg_att = nn.Linear(N_h, 1)
            self.sel_type_att = nn.Linear(N_h, 1)
        self.sel_col_name_enc = nn.LSTM(input_size=N_word+N_h, hidden_size=N_h/2,
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)
        self.sel_out_K = nn.Linear(N_h, N_h)
        self.sel_out_col = nn.Linear(N_h, N_h)
        self.sel_out_K_type = nn.Linear(N_h, N_h)
        self.sel_out_agg = nn.Linear(N_h, N_h)
        self.sel_out = nn.Sequential(nn.Tanh(), nn.Linear(N_h, 1))
        self.softmax = nn.Softmax(dim=1)
        self.agg_out_agg = nn.Linear(N_word, N_h)
        self.x_type_out = nn.Linear(N_word, N_h)
        self.c_type_out = nn.Linear(N_word, N_h)

        
    def forward(self, x_emb_var, x_len, col_inp_var, col_len, col_num, agg_emb_var, x_type_emb_var, col_type_inp_var, gt_agg=None):
        B = len(x_emb_var)
        max_x_len = max(x_len)
        
        chosen_agg_idx = torch.LongTensor(gt_agg)
        aux_range = torch.LongTensor(range(len(gt_agg)))
        if x_emb_var.is_cuda:
            chosen_agg_idx = chosen_agg_idx.cuda()
            aux_range = aux_range.cuda()
            
        #x_type_enc: (B, max_x_len, hid_dim)
        #col_type_enc: (B, max_col_len, hid_dim)
        col_type_enc = self.c_type_out(col_type_inp_var)
        col_emb_concat = torch.cat((col_inp_var, col_type_enc), 2)
        e_col, _ = run_lstm(self.sel_col_name_enc, col_emb_concat, col_len)

        if self.use_ca:
            x_type_enc = self.x_type_out(x_type_emb_var)
            x_emb_concat = torch.cat((x_emb_var, x_type_enc), 2)
            h_enc, _ = run_lstm(self.sel_lstm, x_emb_concat, x_len)
            #att_val: (B, max_col_len, max_x_len)
            att_val = torch.bmm(e_col, self.sel_att(h_enc).transpose(1, 2))
            #chosen_agg_att
            agg_enc = self.agg_out_agg(agg_emb_var)
            #agg_enc: (B, 6, hid_dim)
            #self.sel_att(h_enc) -> (B, max_x_len, hid_dim) .transpose(1, 2) -> (B, hid_dim, max_x_len)
            #att_val_agg: (B, 6, max_x_len)
            att_val_agg = torch.bmm(agg_enc, self.sel_agg_att(h_enc).transpose(1, 2))
            
            for idx, num in enumerate(x_len):
                if num < max_x_len:
                    att_val[idx, :, num:] = -100
            att = self.softmax(att_val.view((-1, max_x_len))).view(B, -1, max_x_len)
            #K_sel_expand -> (B, max_number of col names in batch tables, hid_dim)
            K_sel_expand = (h_enc.unsqueeze(1) * att.unsqueeze(3)).sum(2)            
        else:
            h_enc, _ = run_lstm(self.sel_lstm, x_emb_var, x_len)
            att_val = self.sel_att(h_enc).squeeze()
            for idx, num in enumerate(x_len):
                if num < max_x_len:
                    att_val[idx, num:] = -100
                    
            att = self.softmax(att_val)
            K_sel = (h_enc * att.unsqueeze(2).expand_as(h_enc)).sum(1)
            K_sel_expand=K_sel.unsqueeze(1)
        
        #att_agg: (B, 6, max_x_len)
        att_agg = self.softmax(att_val_agg.view((-1, max_x_len))).view(B, -1, max_x_len)
        #h_enc.unsqueeze(1) -> (B, 1, max_x_len, hid_dim)
        #att_agg.unsqueeze(3) -> (B, 6, max_x_len, 1)
        #K_agg_expand -> (B, 6, hid_dim)
        K_agg_expand = (h_enc.unsqueeze(1) * att_agg.unsqueeze(3)).sum(2)
        #chosen_agg: (B, hid_dim)
        chosen_agg = K_agg_expand[aux_range, chosen_agg_idx]
        
        sel_score = self.sel_out(self.sel_out_K(K_sel_expand) + \
                self.sel_out_col(e_col)).squeeze()
        #+ self.sel_out_agg(chosen_agg.unsqueeze(1).expand_as(K_sel_expand))
        
        max_col_num = max(col_num)
        for idx, num in enumerate(col_num):
            if num < max_col_num:
                sel_score[idx, num:] = -100

        return sel_score
