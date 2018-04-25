import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from net_utils import run_lstm, col_name_encode

class SelCondPredictor(nn.Module):
    def __init__(self, N_word, N_h, N_depth, gpu):
        super(SelCondPredictor, self).__init__()
        self.N_h = N_h
        self.gpu = gpu

        self.selcond_lstm = nn.LSTM(input_size=N_word, hidden_size=N_h/2,
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)
        self.selcond_name_enc = nn.LSTM(input_size=N_word, hidden_size=N_h/2,
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)

        self.sel_num_h = nn.Linear(N_h, N_h)
        self.sel_num_l = nn.Linear(N_h, N_h)
        self.sel_num_out = nn.Sequential(nn.Tanh(), nn.Linear(N_h, 4))

        self.sel_att = nn.Linear(N_h, N_h)
        self.sel_out_K = nn.Linear(N_h, N_h)
        self.sel_out_col = nn.Linear(N_h, N_h)
        self.sel_out = nn.Sequential(nn.Tanh(), nn.Linear(N_h, 1))

        self.num_type_att = nn.Linear(N_h, N_h)
        self.ty_num_out = nn.Linear(N_h, N_h)
        self.cond_num_out = nn.Sequential(nn.Tanh(), nn.Linear(N_h, 5))

        self.agg_num_att = nn.Linear(N_h, N_h)
        self.agg_num_out_K = nn.Linear(N_h, N_h)
        self.agg_num_out_col = nn.Linear(N_h, N_h)
        self.agg_num_out = nn.Sequential(nn.Linear(N_h, N_h), nn.Tanh(),
                nn.Linear(N_h, 4))

        self.agg_op_att = nn.Linear(N_h, N_h)
        self.agg_op_out_K = nn.Linear(N_h, N_h)
        self.agg_op_out_col = nn.Linear(N_h, N_h)
        self.agg_op_out = nn.Sequential(nn.Linear(N_h, N_h), nn.Tanh(),
                nn.Linear(N_h, 6))

        self.gby_num_h = nn.Linear(N_h, N_h)
        self.gby_num_l = nn.Linear(N_h, N_h)
        self.gby_num_out = nn.Sequential(nn.Tanh(), nn.Linear(N_h, 3))

        self.gby_att = nn.Linear(N_h, N_h)
        self.gby_out_K = nn.Linear(N_h, N_h)
        self.gby_out_col = nn.Linear(N_h, N_h)
        self.gby_out = nn.Sequential(nn.Tanh(), nn.Linear(N_h, 1))

        self.ody_num_h = nn.Linear(N_h, N_h)
        self.ody_num_l = nn.Linear(N_h, N_h)
        self.ody_num_out = nn.Sequential(nn.Tanh(), nn.Linear(N_h, 2)) # up to 1 order by columns for now

        self.ody_att = nn.Linear(N_h, N_h)
        self.ody_out_K = nn.Linear(N_h, N_h)
        self.ody_out_col = nn.Linear(N_h, N_h)
        self.ody_out = nn.Sequential(nn.Tanh(), nn.Linear(N_h, 1)) # (B, max_col)

        self.ody_agg_att = nn.Linear(N_h, N_h)
        self.ody_agg_out_K = nn.Linear(N_h, N_h)
        self.ody_agg_out_col = nn.Linear(N_h, N_h)
        self.ody_agg_out = nn.Sequential(nn.Linear(N_h, N_h), nn.Tanh(),
                nn.Linear(N_h, 6))

        self.ody_par_h = nn.Linear(N_h, N_h)
        self.ody_par_l = nn.Linear(N_h, N_h)
        self.ody_par_out = nn.Sequential(nn.Tanh(), nn.Linear(N_h, 3)) # 0, 1, or 2 parity

        self.cond_col_att = nn.Linear(N_h, N_h)
        self.cond_col_out_K = nn.Linear(N_h, N_h)
        self.cond_col_out_col = nn.Linear(N_h, N_h)
        self.cond_col_out_sel = nn.Linear(N_h, N_h)
        self.col_att = nn.Linear(N_h, N_h)
        self.cond_col_out = nn.Sequential(nn.ReLU(), nn.Linear(N_h, 1))

        self.softmax = nn.Softmax() #dim=1


    def forward(self, x_emb_var, x_len, col_inp_var, col_len,  gt_sel):
        max_x_len = max(x_len)
        max_col_len = max(col_len)
        B = len(x_len)

        e_col, _ = run_lstm(self.selcond_name_enc, col_inp_var, col_len)
        h_enc, _ = run_lstm(self.selcond_lstm, x_emb_var, x_len)

        # Predict the number of select columns
        #sel_num_att:(B, max_col_len, max_x_len)
        sel_num_att = torch.bmm(e_col, self.sel_num_h(h_enc).transpose(1, 2))

        for idx, num in enumerate(col_len):
            if num < max_col_len:
                sel_num_att[idx, num:, :] = -100
        for idx, num in enumerate(x_len):
            if num < max_x_len:
                sel_num_att[idx, :, num:] = -100

        #sel_num_att_val: (B, max_col_len, max_x_len)
        sel_num_att_val = self.softmax(sel_num_att.view((-1, max_x_len))).view(B, -1, max_x_len)
        #h_enc.unsqueeze(1): (B, 1, max_x_len, hid_dim)
        #sel_num_att.unsqueeze(3): (B, max_col_len, max_x_len, 1)
        #sel_num_K (B, max_col_len, hid_dim)
        sel_num_K = (h_enc.unsqueeze(1) * sel_num_att.unsqueeze(3)).sum(2).sum(1)
        #sel_num_K (B, 4)
        sel_num_score = self.sel_num_out(self.sel_num_l(sel_num_K))

        #Predict the selection condition
        #att_val: (B, max_col_len, max_x_len)
        sel_att_val = torch.bmm(e_col, self.sel_att(h_enc).transpose(1, 2))
        for idx, num in enumerate(x_len):
            if num < max_x_len:
                sel_att_val[idx, :, num:] = -100
        sel_att = self.softmax(sel_att_val.view((-1, max_x_len))).view(B, -1, max_x_len)
        #K_sel_expand -> (B, max_number of col names in batch tables, hid_dim)
        K_sel_expand = (h_enc.unsqueeze(1) * sel_att.unsqueeze(3)).sum(2)
        sel_score = self.sel_out(self.sel_out_K(K_sel_expand) + \
                self.sel_out_col(e_col)).squeeze()

        for idx, num in enumerate(col_len):
            if num < max_col_len:
                sel_score[idx, num:] = -100

        # Predict the number of aggregators for each select column 
        chosen_sel_gt = []
        if gt_sel is None:
            sel_nums = [x + 1 for x in list(np.argmax(sel_num_score.data.cpu().numpy(), axis=1))]
            # print 'sel_nums', sel_nums
            sel_col_scores = sel_score.data.cpu().numpy()
            chosen_sel_gt = [list(np.argsort(-sel_col_scores[b])[:sel_nums[b]])
                    for b in range(len(sel_nums))]
        else:
            # chosen_sel_gt = gt_sel
            chosen_sel_gt = []
            for x in gt_sel:
                curr = x[0]
                curr_sel = [curr]
                for col in x:
                    if col != curr:
                        curr_sel.append(col)
                chosen_sel_gt.append(curr_sel)

        col_emb = []
        for b in range(B):
            cur_col_emb = torch.stack([e_col[b, x]
                for x in chosen_sel_gt[b]] + [e_col[b, 0]] *
                (4 - len(chosen_sel_gt[b])))  # Pad the columns to maximum (4)
            col_emb.append(cur_col_emb)
            # print list(cur_col_emb.size())

        col_emb = torch.stack(col_emb)

        # predict the num aggs for each sel col
        agg_num_att_val = torch.matmul(self.agg_num_att(h_enc).unsqueeze(1),
                col_emb.unsqueeze(3)).squeeze()
        for idx, num in enumerate(x_len):
            if num < max_x_len:
                agg_num_att_val[idx, :, num:] = -100
        agg_num_att = self.softmax(agg_num_att_val.view(B*4, -1)).view(B, 4, -1)
        K_agg_num = (h_enc.unsqueeze(1) * agg_num_att.unsqueeze(3)).sum(2)

        agg_num_score = self.agg_num_out(self.agg_num_out_K(K_agg_num) +
                self.agg_num_out_col(col_emb)).squeeze()

        # predict the aggregation operators
        agg_op_att_val = torch.matmul(self.agg_op_att(h_enc).unsqueeze(1),
                col_emb.unsqueeze(3)).squeeze()
        for idx, num in enumerate(x_len):
            if num < max_x_len:
                agg_op_att_val[idx, :, num:] = -100
        agg_op_att = self.softmax(agg_op_att_val.view(B*4, -1)).view(B, 4, -1)
        K_agg_op = (h_enc.unsqueeze(1) * agg_op_att.unsqueeze(3)).sum(2)

        agg_op_score = self.agg_op_out(self.agg_op_out_K(K_agg_op) +
                self.agg_op_out_col(col_emb)).squeeze()

        ###
        # Group by 
        # Predict the number of group by columns
        gby_num_att = torch.bmm(e_col, self.gby_num_h(h_enc).transpose(1, 2))

        for idx, num in enumerate(col_len):
            if num < max_col_len:
                gby_num_att[idx, num:, :] = -100
        for idx, num in enumerate(x_len):
            if num < max_x_len:
                gby_num_att[idx, :, num:] = -100

        gby_num_att_val = self.softmax(gby_num_att.view((-1, max_x_len))).view(B, -1, max_x_len)
        gby_num_K = (h_enc.unsqueeze(1) * gby_num_att.unsqueeze(3)).sum(2).sum(1)
        gby_num_score = self.gby_num_out(self.gby_num_l(gby_num_K))

        # Predict the group by columns 
        gby_att_val = torch.bmm(e_col, self.gby_att(h_enc).transpose(1, 2))
        for idx, num in enumerate(x_len):
            if num < max_x_len:
                gby_att_val[idx, :, num:] = -100
        gby_att = self.softmax(gby_att_val.view((-1, max_x_len))).view(B, -1, max_x_len)
        K_gby_expand = (h_enc.unsqueeze(1) * gby_att.unsqueeze(3)).sum(2)
        gby_score = self.gby_out(self.gby_out_K(K_gby_expand) + \
                self.gby_out_col(e_col)).squeeze()

        ###
        # Order by
        # Predict the number of order by columns
        ody_num_att = torch.bmm(e_col, self.ody_num_h(h_enc).transpose(1, 2))

        for idx, num in enumerate(col_len):
            if num < max_col_len:
                ody_num_att[idx, num:, :] = -100
        for idx, num in enumerate(x_len):
            if num < max_x_len:
                ody_num_att[idx, :, num:] = -100

        ody_num_att_val = self.softmax(ody_num_att.view((-1, max_x_len))).view(B, -1, max_x_len)
        ody_num_K = (h_enc.unsqueeze(1) * ody_num_att.unsqueeze(3)).sum(2).sum(1)
        ody_num_score = self.ody_num_out(self.ody_num_l(ody_num_K))

        # Predict the order by columns 
        ody_att_val = torch.bmm(e_col, self.ody_att(h_enc).transpose(1, 2))
        for idx, num in enumerate(x_len):
            if num < max_x_len:
                ody_att_val[idx, :, num:] = -100
        ody_att = self.softmax(ody_att_val.view((-1, max_x_len))).view(B, -1, max_x_len)
        K_ody_expand = (h_enc.unsqueeze(1) * ody_att.unsqueeze(3)).sum(2)
        ody_score = self.ody_out(self.ody_out_K(K_ody_expand) + \
                self.ody_out_col(e_col)).squeeze()

        # Predict the agg for each column
        ody_agg_att_val = torch.matmul(self.ody_agg_att(h_enc).unsqueeze(1),
                col_emb.unsqueeze(3)).squeeze()
        for idx, num in enumerate(x_len):
            if num < max_x_len:
                ody_agg_att_val[idx, :, num:] = -100
        ody_agg_att = self.softmax(ody_agg_att_val.view(B*4, -1)).view(B, 4, -1)
        K_ody_agg = (h_enc.unsqueeze(1) * ody_agg_att.unsqueeze(3)).sum(2)

        ody_agg_score = self.ody_agg_out(self.ody_agg_out_K(K_ody_agg) +
                self.ody_agg_out_col(col_emb)).squeeze()        

        # Predict the parity

        ody_par_att = torch.bmm(e_col, self.ody_par_h(h_enc).transpose(1, 2))

        for idx, num in enumerate(col_len):
            if num < max_col_len:
                ody_par_att[idx, num:, :] = -100
        for idx, num in enumerate(x_len):
            if num < max_x_len:
                ody_par_att[idx, :, num:] = -100

        ody_par_att_val = self.softmax(ody_par_att.view((-1, max_x_len))).view(B, -1, max_x_len)
        ody_par_K = (h_enc.unsqueeze(1) * ody_par_att.unsqueeze(3)).sum(2).sum(1)
        ody_par_score = self.ody_par_out(self.ody_par_l(ody_par_K))

        ########### Where condition
        # Predict the number of conditions
        #att_num_type_val:(B, max_col_len, max_x_len)
        att_num_type_val = torch.bmm(e_col, self.num_type_att(h_enc).transpose(1, 2))

        for idx, num in enumerate(col_len):
            if num < max_col_len:
                att_num_type_val[idx, num:, :] = -100
        for idx, num in enumerate(x_len):
            if num < max_x_len:
                att_num_type_val[idx, :, num:] = -100

        #att_num_type: (B, max_col_len, max_x_len)
        att_num_type = self.softmax(att_num_type_val.view((-1, max_x_len))).view(B, -1, max_x_len)
        #h_enc.unsqueeze(1): (B, 1, max_x_len, hid_dim)
        #att_num_type.unsqueeze(3): (B, max_col_len, max_x_len, 1)
        #K_num_type (B, max_col_len, hid_dim)
        K_num_type = (h_enc.unsqueeze(1) * att_num_type.unsqueeze(3)).sum(2).sum(1)
        #K_cond_num: (B, hid_dim)
        #K_num_type (B, hid_dim)
        cond_num_score = self.cond_num_out(self.ty_num_out(K_num_type))

        #Predict the columns of conditions
        if gt_sel is None:
            gt_sel = chosen_sel_gt#np.argmax(sel_score.data.cpu().numpy(), axis=1)
        #gt_sel (B)
        # print gt_sel
        temp_gt_sel = [x[0] for x in gt_sel]
        chosen_sel_idx = torch.LongTensor(temp_gt_sel)#torch.LongTensor(gt_sel)
        #aux_range (B) (0,1,...)
        aux_range = torch.LongTensor(range(len(gt_sel)))
        if x_emb_var.is_cuda:
            chosen_sel_idx = chosen_sel_idx.cuda()
            aux_range = aux_range.cuda()
        #chosen_e_col: (B, hid_dim)

        chosen_e_col = e_col[aux_range, chosen_sel_idx]
        #chosen_e_col.unsqueeze(2): (B, hid_dim, 1)
        #self.col_att(h_enc): (B, max_x_len, hid_dim)
        #att_sel_val: (B, max_x_len)
        att_sel_val = torch.bmm(self.col_att(h_enc), chosen_e_col.unsqueeze(2)).squeeze()

        col_att_val = torch.bmm(e_col, self.cond_col_att(h_enc).transpose(1, 2))
        for idx, num in enumerate(x_len):
            if num < max_x_len:
                col_att_val[idx, :, num:] = -100
                att_sel_val[idx, num:] = -100
        sel_att = self.softmax(att_sel_val)
        K_sel_agg = (h_enc * sel_att.unsqueeze(2).expand_as(h_enc)).sum(1)
        col_att = self.softmax(col_att_val.view((-1, max_x_len))).view(B, -1, max_x_len)
        K_cond_col = (h_enc.unsqueeze(1) * col_att.unsqueeze(3)).sum(2)

        cond_col_score = self.cond_col_out(self.cond_col_out_K(K_cond_col)
                + self.cond_col_out_col(e_col)
                + self.cond_col_out_sel(K_sel_agg.unsqueeze(1).expand_as(K_cond_col))).squeeze()

        for b, num in enumerate(col_len):
            if num < max_col_len:
                cond_col_score[b, num:] = -100

        # cond_num_score = None
        # cond_col_score = None

        sel_cond_score = (sel_num_score, cond_num_score, sel_score, cond_col_score, agg_num_score, agg_op_score,
                             gby_num_score, gby_score, ody_num_score, ody_score, ody_agg_score, ody_par_score)

        return sel_cond_score
