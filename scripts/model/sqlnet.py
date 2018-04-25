import json
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from modules.word_embedding import WordEmbedding
from modules.aggregator_predict import AggPredictor
from modules.sel_condition_predict import SelCondPredictor
from modules.condtion_op_str_predict import CondOpStrPredictor


class SQLNet(nn.Module):
    def __init__(self, word_emb, N_word, N_h=120, N_depth=2,
            gpu=False, trainable_emb=False):
        super(SQLNet, self).__init__()
        self.trainable_emb = trainable_emb

        self.gpu = gpu
        self.N_h = N_h
        self.N_depth = N_depth

        self.max_col_num = 45
        self.max_tok_num = 200
        self.SQL_TOK = ['<UNK>', '<END>', 'WHERE', 'AND',
                'EQL', 'GT', 'LT', '<BEG>']
        self.COND_OPS = ['EQL', 'GT', 'LT']

        self.embed_layer = WordEmbedding(word_emb, N_word, gpu,
                self.SQL_TOK, trainable=trainable_emb)

        #Predict aggregator
        self.agg_pred = AggPredictor(N_word, N_h, N_depth)

        #Predict select column + condition number and columns
        self.selcond_pred = SelCondPredictor(N_word, N_h, N_depth, gpu)

        #Predict condition operators and string values
        self.op_str_pred = CondOpStrPredictor(N_word, N_h, N_depth,
                self.max_col_num, self.max_tok_num, gpu)

        self.CE = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax()
        self.log_softmax = nn.LogSoftmax()
        self.bce_logit = nn.BCEWithLogitsLoss()
        self.sigm = nn.Sigmoid()
        if gpu:
            self.cuda()


    def get_str_index(self, all_toks, this_str):
        cur_seq = []
        tok_gt_1 = [t for t in all_toks if len(t) > 1]
        if this_str in all_toks:
            all_str = ['<BEG>', this_str, '<END>']
            cur_seq = [all_toks.index(s) if s in all_toks else 0 for s in all_str]
        elif len(tok_gt_1) > 0:
            flag = False
            for tgt in tok_gt_1:
                if set(tgt).issubset(this_str):
                    not_tgt = [x for x in this_str if x not in tgt]
                    if len(not_tgt) > 0:
                        not_tgt = [[x] for x in not_tgt]
                        all_str = [tgt] + not_tgt
                    else:
                        all_str = [tgt]
                    beg_ind = all_toks.index('<BEG>') if '<BEG>' in all_toks else 0
                    end_ind = all_toks.index('<END>') if '<END>' in all_toks else 0
                    cur_seq = sorted([all_toks.index(s) if s in all_toks else 0 for s in all_str])
                    cur_seq = [beg_ind] + cur_seq + [end_ind]
                elif set(this_str).issubset(tgt):
                    all_str = ['<BEG>', tgt, '<END>']
                    cur_seq = [all_toks.index(s) if s in all_toks else 0 for s in all_str]

                if len(cur_seq) > 0:
                    flag = True
                    break

            if not flag:
                all_str = ['<BEG>'] + [[x] for x in this_str] + ['<END>']
                cur_seq = [all_toks.index(s) if s in all_toks else 0 for s in all_str]
        else:
            all_str = ['<BEG>'] + [[x] for x in this_str] + ['<END>']
            cur_seq = [all_toks.index(s) if s in all_toks else 0 for s in all_str]

        return cur_seq


    def generate_gt_where_seq(self, q, col, query):
        """
        cur_seq is the indexes (in question toks) of string value in each where cond
        """
        ret_seq = []
        for cur_q, cur_col, cur_query in zip(q, col, query):
            cur_values = []
            st = cur_query.index(u'WHERE')+1 if \
                    u'WHERE' in cur_query else len(cur_query)
            all_toks = ['<BEG>'] + cur_q + ['<END>']
            while st < len(cur_query):
                ed = len(cur_query) if 'AND' not in cur_query[st:]\
                        else cur_query[st:].index('AND') + st
                if 'EQL' in cur_query[st:ed]:
                    op = cur_query[st:ed].index('EQL') + st
                elif 'GT' in cur_query[st:ed]:
                    op = cur_query[st:ed].index('GT') + st
                elif 'LT' in cur_query[st:ed]:
                    op = cur_query[st:ed].index('LT') + st
                else:
                    raise RuntimeError("No operator in it!")

                this_str = cur_query[op+1:ed]
                cur_seq = self.get_str_index(all_toks, this_str)
                cur_values.append(cur_seq)
                st = ed+1
            ret_seq.append(cur_values)
        return ret_seq


    def forward(self, q, col, col_num, pred_entry,
            gt_where = None, gt_cond=None, gt_sel=None):
        B = len(q)
        pred_agg, pred_sel, pred_cond = pred_entry

        agg_score = None
        sel_cond_score = None
        cond_op_str_score = None

        #Predict aggregator
        if self.trainable_emb:
            if pred_agg:
                x_emb_var, x_len = self.agg_embed_layer.gen_x_batch(q, col)
                col_inp_var, col_name_len, col_len = \
                        self.agg_embed_layer.gen_col_batch(col)
                max_x_len = max(x_len)
                agg_score = self.agg_pred(x_emb_var, x_len, col_inp_var,
                        col_name_len, col_len, col_num, gt_sel=gt_sel)

            if pred_sel:
                x_emb_var, x_len = self.sel_embed_layer.gen_x_batch(q, col)
                col_inp_var, col_name_len, col_len = \
                        self.sel_embed_layer.gen_col_batch(col)
                max_x_len = max(x_len)
                # print 'trainable emb'
                sel_score = self.selcond_pred(x_emb_var, x_len, col_inp_var,
                        col_name_len, col_len, col_num, gt_sel = gt_sel) #temp for agg num testing

            if pred_cond:
                x_emb_var, x_len = self.cond_embed_layer.gen_x_batch(q, col)
                col_inp_var, col_name_len, col_len = \
                        self.cond_embed_layer.gen_col_batch(col)
                max_x_len = max(x_len)
                cond_score = self.cond_pred(x_emb_var, x_len, col_inp_var,
                        col_name_len, col_len, col_num,
                        gt_where, gt_cond)
        else:
            x_emb_var, x_len = self.embed_layer.gen_x_batch(q, col, is_q=True)
            col_inp_var, col_len = self.embed_layer.gen_x_batch(col, col, is_list=True)
            agg_emb_var = self.embed_layer.gen_agg_batch(q)
            max_x_len = max(x_len)
            if pred_agg:
                agg_score = self.agg_pred(x_emb_var, x_len, agg_emb_var, col_inp_var, col_len)

            if pred_sel:
                # print 'not trainable'
                sel_cond_score = self.selcond_pred(x_emb_var, x_len, col_inp_var, col_len, gt_sel)

            if pred_cond:
                cond_op_str_score = self.op_str_pred(x_emb_var, x_len, col_inp_var, col_len,
                                                     gt_where, gt_cond, sel_cond_score)

        return (agg_score, sel_cond_score, cond_op_str_score)


    def loss(self, score, truth_num, pred_entry, gt_where):
        pred_agg, pred_sel, pred_cond = pred_entry
        agg_score, sel_cond_score, cond_op_str_score = score

        sel_num_score, cond_num_score, sel_score, cond_col_score, agg_num_score, agg_op_score, gby_num_score, gby_score, ody_num_score, ody_score, ody_agg_score, ody_par_score = sel_cond_score

        B = len(truth_num)
        loss = 0
        if pred_agg:
            # # of agg loss
            for b in range(len(truth_num)):
                curr_col = truth_num[b][1][0]
                curr_col_num_aggs = 0
                gt_aggs_num = []
                for i, col in enumerate(truth_num[b][1]):
                    if col != curr_col:
                        gt_aggs_num.append(curr_col_num_aggs)
                        curr_col = col
                        curr_col_num_aggs = 0
                    if truth_num[b][0][i] != 0:
                        curr_col_num_aggs += 1
                gt_aggs_num.append(curr_col_num_aggs)
                # print gt_aggs_num
                data = torch.from_numpy(np.array(gt_aggs_num)) #supposed to be gt # of aggs
                if self.gpu:
                    agg_num_truth_var = Variable(data.cuda())
                else:
                    agg_num_truth_var = Variable(data)
                agg_num_pred = agg_num_score[b, :truth_num[b][5]] # supposed to be gt # of select columns
                loss += (self.CE(agg_num_pred, agg_num_truth_var) \
                        / len(truth_num))

                # agg prediction loss
                T = 6 #num agg ops 
                truth_prob = np.zeros((truth_num[b][5], T), dtype=np.float32) 
                gt_agg_by_sel = []
                curr_sel_aggs = []
                curr_col = truth_num[b][1][0]
                col_counter = 0
                for i, col in enumerate(truth_num[b][1]):
                    if col != curr_col:
                        gt_agg_by_sel.append(curr_sel_aggs)
                        curr_col = col
                        col_counter += 1
                        curr_sel_aggs = [truth_num[b][0][i]]
                        truth_prob[col_counter][curr_sel_aggs] = 1
                    else:
                        curr_sel_aggs.append(truth_num[b][0][i])    
                        truth_prob[col_counter][curr_sel_aggs] = 1
                data = torch.from_numpy(truth_prob)
                if self.gpu:
                    agg_op_truth_var = Variable(data.cuda())
                else:
                    agg_op_truth_var = Variable(data)
                # print truth_num[b][0], truth_num[b][1]
                # print truth_prob
                # print (truth_num[b][5])
                agg_op_prob = self.sigm(agg_op_score[b, :truth_num[b][5]])
                agg_bce_loss = -torch.mean( 3*(agg_op_truth_var * \
                        torch.log(agg_op_prob+1e-10)) + \
                        (1-agg_op_truth_var) * torch.log(1-agg_op_prob+1e-10) )
                loss += agg_bce_loss / len(truth_num)

                # for i in len(truth_num[b][5]):
                #     data = torch.from_numpy(np.array(truth_num[b][1]))
                #     if self.gpu:
                #         agg_op_truth_var = Variable(data.cuda())
                #     else:
                #         agg_op_truth_var = Variable(data)
                # agg_op_pred = agg_op_score[b, :len(truth_num[b][5])]
                # loss += (self.CE(agg_op_pred, agg_op_truth_var) \
                #         / len(truth_num))

            # agg_truth = map(lambda x:x[0], truth_num)
            # data = torch.from_numpy(np.array(agg_truth))
            # if self.gpu:
            #     agg_truth_var = Variable(data.cuda())
            # else:
            #     agg_truth_var = Variable(data)

            # loss += self.CE(agg_score, agg_truth_var)

        if pred_sel:
            #Evaluate the number of select columns
            sel_num_truth = map(lambda x: x[5]-1, truth_num) #might need to be the length of the set of columms 
            data = torch.from_numpy(np.array(sel_num_truth))
            if self.gpu:
                sel_num_truth_var = Variable(data.cuda())
            else:
                sel_num_truth_var = Variable(data)
            loss += self.CE(sel_num_score, sel_num_truth_var)

            # Evaluate the select columns
            T = len(sel_score[0]) 
            truth_prob = np.zeros((B, T), dtype=np.float32)
            for b in range(B):                          
                truth_prob[b][truth_num[b][1]] = 1
            data = torch.from_numpy(truth_prob)
            if self.gpu:
                sel_col_truth_var = Variable(data.cuda())
            else:
                sel_col_truth_var = Variable(data)
            sel_col_prob = self.sigm(sel_score)
            sel_bce_loss = -torch.mean( 3*(sel_col_truth_var * \
                    torch.log(sel_col_prob+1e-10)) + \
                    (1-sel_col_truth_var) * torch.log(1-sel_col_prob+1e-10) )
            loss += sel_bce_loss

            # Evaluate the number of group by columns 
            gby_num_truth = map(lambda x: x[7], truth_num) 
            data = torch.from_numpy(np.array(gby_num_truth))
            if self.gpu:
                gby_num_truth_var = Variable(data.cuda())
            else:
                gby_num_truth_var = Variable(data)
            loss += self.CE(gby_num_score, gby_num_truth_var)

            # Evaluate the group by columns
            T = len(gby_score[0]) 
            truth_prob = np.zeros((B, T), dtype=np.float32)
            for b in range(B):                   
                if len(truth_num[b][6]) > 0:
                    truth_prob[b][list(truth_num[b][6])] = 1       
            data = torch.from_numpy(truth_prob)
            if self.gpu:
                gby_col_truth_var = Variable(data.cuda())
            else:
                gby_col_truth_var = Variable(data)
            gby_col_prob = self.sigm(gby_score)
            gby_bce_loss = -torch.mean( 3*(gby_col_truth_var * \
                    torch.log(gby_col_prob+1e-10)) + \
                    (1-gby_col_truth_var) * torch.log(1-gby_col_prob+1e-10) )
            loss += gby_bce_loss

            # Evaluate the number of order by columns 
            ody_num_truth = map(lambda x: x[10], truth_num)  
            data = torch.from_numpy(np.array(ody_num_truth))
            if self.gpu:
                ody_num_truth_var = Variable(data.cuda())
            else:
                ody_num_truth_var = Variable(data)
            loss += self.CE(ody_num_score, ody_num_truth_var)

            # Evaluate the order by columns
            T = len(ody_score[0]) 
            truth_prob = np.zeros((B, T), dtype=np.float32)
            for b in range(B):                   
                if len(truth_num[b][9]) > 0:     
                    truth_prob[b][list(truth_num[b][9])] = 1       
            data = torch.from_numpy(truth_prob)
            if self.gpu:
                ody_col_truth_var = Variable(data.cuda())
            else:
                ody_col_truth_var = Variable(data)
            ody_col_prob = self.sigm(ody_score)
            ody_bce_loss = -torch.mean( 3*(ody_col_truth_var * \
                    torch.log(ody_col_prob+1e-10)) + \
                    (1-ody_col_truth_var) * torch.log(1-ody_col_prob+1e-10) )
            loss += ody_bce_loss

            # # Evaluate agg pred for each
            T = 6 #num agg ops 
            for b in range(B):
                if len(truth_num[b][9]) > 0: 
                    # print 'has count', truth_num[b][8]
                    truth_prob = np.zeros((truth_num[b][10], T), dtype=np.float32) # num order by columns
                    # gt_agg_by_sel = []
                    curr_ody_aggs = []
                    curr_col = truth_num[b][9][0]
                    col_counter = 0
                    for i, col in enumerate(truth_num[b][9]): # loop over order by columns 
                        if col != curr_col:
                            # gt_agg_by_sel.append(curr_ody_aggs)
                            curr_col = col
                            # print curr_ody_aggs
                            truth_prob[col_counter][curr_ody_aggs] = 1
                            curr_ody_aggs = [truth_num[b][8][i]]
                            col_counter += 1
                        else:
                            curr_ody_aggs.append(truth_num[b][8][i])    
                            truth_prob[col_counter][curr_ody_aggs] = 1
                    # print truth_prob
                    data = torch.from_numpy(truth_prob)
                    if self.gpu:
                        ody_agg_truth_var = Variable(data.cuda())
                    else:
                        ody_agg_truth_var = Variable(data)
                    ody_agg_prob = self.sigm(ody_agg_score[b, :truth_num[b][10]])
                    ody_agg_bce_loss = -torch.mean( 5*(ody_agg_truth_var * \
                            torch.log(ody_agg_prob+1e-10)) + \
                            (1-ody_agg_truth_var) * torch.log(1-ody_agg_prob+1e-10) )
                    # print truth_num[b][8]
                    # print ody_agg_bce_loss
                    loss += ody_agg_bce_loss / len(truth_num)

            # Evaluate parity
            ody_par_truth = map(lambda x: x[11] + 1, truth_num)  
            data = torch.from_numpy(np.array(ody_par_truth))
            if self.gpu:
                ody_par_truth_var = Variable(data.cuda())
            else:
                ody_par_truth_var = Variable(data)
            loss += self.CE(ody_par_score, ody_par_truth_var)

        if pred_cond:
            cond_op_score, cond_str_score = cond_op_str_score
            #Evaluate the number of conditions
            cond_num_truth = map(lambda x:x[2], truth_num)
            data = torch.from_numpy(np.array(cond_num_truth))
            if self.gpu:
                cond_num_truth_var = Variable(data.cuda())
            else:
                cond_num_truth_var = Variable(data)
            loss += self.CE(cond_num_score, cond_num_truth_var)

            #Evaluate the columns of conditions
            T = len(cond_col_score[0])
            truth_prob = np.zeros((B, T), dtype=np.float32)
            for b in range(B):
                if len(truth_num[b][3]) > 0:
                    truth_prob[b][list(truth_num[b][3])] = 1
            data = torch.from_numpy(truth_prob)
            if self.gpu:
                cond_col_truth_var = Variable(data.cuda())
            else:
                cond_col_truth_var = Variable(data)

            cond_col_prob = self.sigm(cond_col_score)
            bce_loss = -torch.mean( 3*(cond_col_truth_var * \
                    torch.log(cond_col_prob+1e-10)) + \
                    (1-cond_col_truth_var) * torch.log(1-cond_col_prob+1e-10) )
            loss += bce_loss

            #Evaluate the operator of conditions
            for b in range(len(truth_num)):
                if len(truth_num[b][4]) == 0:
                    continue
                data = torch.from_numpy(np.array(truth_num[b][4]))
                if self.gpu:
                    cond_op_truth_var = Variable(data.cuda())
                else:
                    cond_op_truth_var = Variable(data)
                cond_op_pred = cond_op_score[b, :len(truth_num[b][4])]
                # print 'cond_op_truth_var', list(cond_op_truth_var.size())
                # print 'cond_op_pred', list(cond_op_pred.size())
                loss += (self.CE(cond_op_pred, cond_op_truth_var) \
                        / len(truth_num))

            #Evaluate the strings of conditions
            # for b in range(len(gt_where)):
            #     for idx in range(len(gt_where[b])):
            #         cond_str_truth = gt_where[b][idx]
            #         if len(cond_str_truth) == 1:
            #             continue
            #         data = torch.from_numpy(np.array(cond_str_truth[1:]))
            #         if self.gpu:
            #             cond_str_truth_var = Variable(data.cuda())
            #         else:
            #             cond_str_truth_var = Variable(data)
            #         str_end = len(cond_str_truth)-1
            #         cond_str_pred = cond_str_score[b, idx, :str_end]
            #         loss += (self.CE(cond_str_pred, cond_str_truth_var) \
            #                 / (len(gt_where) * len(gt_where[b])))

        return loss


    def check_acc(self, vis_info, pred_queries, gt_queries, pred_entry, error_print=False):
        def pretty_print(vis_data, pred_query, gt_query):
            print "\n----------detailed error prints-----------"
            try:
                print 'question: ', vis_data[0]
                print 'question_tok: ', vis_data[3]
                print 'headers: (%s)'%(' || '.join(vis_data[1]))
                print 'query:', vis_data[2]
                print "target query: ", gt_query
                print "pred query: ", pred_query
            except:
                print "\n------skipping print: decoding problem ----------------------"

        def gen_cond_str(conds, header):
            if len(conds) == 0:
                return 'None'
            cond_str = []
            for cond in conds:
                cond_str.append(header[cond[0]] + ' ' +
                    self.COND_OPS[cond[1]] + ' ' + unicode(cond[2]).lower())
            return 'WHERE ' + ' AND '.join(cond_str)

        pred_agg, pred_sel, pred_cond = pred_entry

        B = len(gt_queries)

        tot_err = agg_err = sel_num_err = sel_err = cond_err = 0.0
        cond_num_err = cond_col_err = cond_op_err = cond_val_err = 0.0
        agg_num_err = gby_num_err = gby_err = 0.0
        ody_num_cols_err = ody_err = ody_agg_err = ody_par_err = 0.0
        agg_ops = ['None', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
        for b, (pred_qry, gt_qry, vis_data) in enumerate(zip(pred_queries, gt_queries, vis_info)):
            good = True
            

            if pred_sel:
                agg_flag = True
                sel_gt = gt_qry['sel']
                sel_num_gt = len(sel_gt)
                sel_pred = pred_qry['sel']
                sel_num_pred = pred_qry['sel_num'] + 1

                # print sel_num_pred, sel_num_gt
                if sel_num_pred != len(set(sel_gt)):
                    sel_num_err += 1
                    good = False
                    # print 'predicted num:', sel_num_pred, 'gt num:', sel_num_gt
                    # print 'predicted columns:', sel_pred, 'gt columns:', sel_gt, '----------------'

                if sorted(set(sel_pred)) != sorted(set(sel_gt)):
                    sel_err += 1
                    good = False
                    agg_flag = False
                    # print 'predicted columns:', sel_pred, 'gt columns:', sel_gt

                gby_gt = gt_qry['group'][:-1]
                gby_pred = pred_qry['group']
                gby_num_pred = pred_qry['gby_num']
                gby_num_gt = len(gby_gt)
                if gby_num_pred != gby_num_gt:
                    gby_num_err += 1
                    good = False
                    # print 'predicted # gby:', gby_num_pred, 'gt # gby:', gby_num_gt

                if sorted(gby_pred) != sorted(gby_gt):
                    gby_err += 1
                    good = False
                    # print 'predicted gby:', gby_pred, 'gt gby:', gby_gt

                ody_gt_aggs = gt_qry['order'][0]
                ody_gt_cols = gt_qry['order'][1]
                ody_gt_par = gt_qry['order'][2]

                # if len(ody_gt_cols) == 0:
                #     ody_gt_num_cols = []
                # else:
                #     curr_col = ody_gt_cols[0]
                #     curr_col_num_aggs = 0
                #     ody_gt_num_aggs = []
                #     gt_ody_order = [curr_col]
                #     for i, col in enumerate(ody_gt_cols):
                #         if col != curr_col:
                #             gt_ody_order.append(col)
                #             ody_gt_num_aggs.append(curr_col_num_aggs)
                #             curr_col = col
                #             curr_col_num_aggs = 0
                #         if ody_gt_aggs[i] != 0:
                #             curr_col_num_aggs += 1
                #     ody_gt_num_aggs.append(curr_col_num_aggs)

                ody_num_cols_pred = pred_qry['ody_num']
                ody_cols_pred = pred_qry['order']
                ody_aggs_pred = pred_qry['ody_agg']
                ody_par_pred = pred_qry['parity']

                if ody_num_cols_pred != len(ody_gt_cols):
                    ody_num_cols_err += 1
                    # print 'predicted #: ', ody_num_cols_pred, 'gold #:', len(ody_gt_cols)

                if ody_cols_pred != ody_gt_cols:
                    ody_err += 1
                    # print 'predicted cols:', ody_cols_pred, 'gold cols:', ody_gt_cols


                if ody_aggs_pred != ody_gt_aggs:
                    ody_agg_err += 1
                    # print 'predicted aggs:', ody_aggs_pred, 'gold aggs:', ody_gt_aggs

                if ody_par_pred != ody_gt_par:
                    ody_par_err += 1
                    # print 'predicted par:', ody_par_pred, 'gold par:', ody_gt_par

            if pred_agg:
                agg_gt = gt_qry['agg']

                curr_col = gt_qry['sel'][0]
                curr_col_num_aggs = 0
                gt_aggs_num = []
                gt_sel_order = [curr_col]
                for i, col in enumerate(gt_qry['sel']):
                    if col != curr_col:
                        gt_sel_order.append(col)
                        gt_aggs_num.append(curr_col_num_aggs)
                        curr_col = col
                        curr_col_num_aggs = 0
                    if agg_gt[i] != 0:
                        curr_col_num_aggs += 1
                gt_aggs_num.append(curr_col_num_aggs)

                if pred_qry['agg_num'] != gt_aggs_num:
                    agg_num_err += 1
                    # print 'predicted #:', pred_qry['agg_num'], 'gt #:', gt_aggs_num 

                # for idx in range(len(gt_sel_order)):
                #     curr_col_gt_aggs = [x for i, x in enumerate(gt_qry['agg']) if gt_qry['sel'][i] == gt_sel_order[idx]]
                #     curr_col_pred_aggs = [x for i, x in enumerate(pred_qry['agg']) if pred_qry['sel'][i] == gt_sel_order[idx]]
                #     if sorted(curr_col_pred_aggs) != sorted(curr_col_gt_aggs):
                #         agg_err += 1
                #         print 'predicted ops:', pred_qry['agg'], 'gt ops:', gt_qry['agg']
                #         break

                if sorted(pred_qry['agg']) != sorted(gt_qry['agg']): # naive
                    agg_err += 1
                    # print 'predicted ops:', pred_qry['agg'], 'gt ops:', gt_qry['agg']


                # for idx in range(len(gt_sel_order)):
                #     if not agg_flag:
                #         break
                #     gt_idx = tuple(x for x in gt_sel_order).index(pred_qry['sel'][idx]) # from gt order to predicted (pred_sel)
                #     if pred_qry['agg_num'][idx] != gt_aggs_num[gt_idx]:
                #         agg_num_err += 1
                #         agg_flag = False
    
                # print 'predicted #:', pred_qry['agg_num'], 'gt #:', gt_aggs_num
                # print 'predicted cols:', pred_qry['sel'], 'gt cols:', gt_sel_order

                # agg_pred = pred_qry['agg']
                # agg_gt = gt_qry['agg']
                # if agg_pred != agg_gt:
                #     agg_err += 1
                #     good = False

            if pred_cond:
                cond_pred = pred_qry['conds']
                cond_gt = gt_qry['cond']#[x[1:] for x in gt_qry['cond']]
                flag = True
                if len(cond_pred) != len(cond_gt):
                    flag = False
                    cond_num_err += 1
                    # print 'predicted #:', len(cond_pred), 'gt op:', len(cond_gt)

                if flag and set(x[0] for x in cond_pred) != \
                        set(x[0] for x in cond_gt):
                    flag = False
                    cond_col_err += 1
                    # print 'predicted col:', [x[0] for x in cond_pred], 'gt col:', [x[0] for x in cond_gt]
                    # print cond_gt, cond_pred

                for idx in range(len(cond_pred)):
                    if not flag:
                        break
                    gt_idx = tuple(x[0] for x in cond_gt).index(cond_pred[idx][0])
                    if flag and cond_gt[gt_idx][1] != cond_pred[idx][1]:
                        flag = False
                        cond_op_err += 1
                        # print 'predicted op:', [x[1] for x in cond_pred], 'gt op:', [x[1] for x in cond_gt]
                # for idx in range(len(cond_pred)):
                #     if not flag:
                #         break
                #     gt_idx = tuple(
                #             x[0] for x in cond_gt).index(cond_pred[idx][0])
                #     if flag and unicode(cond_gt[gt_idx][2]).lower() != \
                #             unicode(cond_pred[idx][2]).lower():
                #         flag = False
                #         cond_val_err += 1

                if not flag:
                    cond_err += 1
                    good = False

            if not good:
                if error_print:
                    pretty_print(vis_data, pred_qry, gt_qry)
                tot_err += 1

        return np.array((agg_num_err, agg_err, sel_err, cond_err, sel_num_err, cond_num_err, cond_col_err, cond_op_err, cond_val_err, gby_num_err, gby_err, ody_num_cols_err, ody_err, ody_agg_err, ody_par_err)), tot_err


    def gen_query(self, score, q, col, raw_q, raw_col,
            pred_entry, verbose=False, gt_sel = None, gt_ody = None):
        def merge_tokens(tok_list, raw_tok_str):
            """
            tok_list: list of string words in current cond
            raw_tok_str: list of words in question
            """
            tok_str = raw_tok_str.lower()
            alphabet = 'abcdefghijklmnopqrstuvwxyz0123456789$('
            special = {'-LRB-':'(',
                    '-RRB-':')',
                    '-LSB-':'[',
                    '-RSB-':']',
                    '``':'"',
                    '\'\'':'"',
                    '--':u'\u2013'}
            ret = ''
            double_quote_appear = 0
            tok_list = [x for gx in tok_list for x in gx]
            for raw_tok in tok_list:
                if not raw_tok:
                    continue
                tok = special.get(raw_tok, raw_tok)
                if tok == '"':
                    double_quote_appear = 1 - double_quote_appear

                if len(ret) == 0:
                    pass
                elif len(ret) > 0 and ret + ' ' + tok in tok_str:
                    ret = ret + ' '
                elif len(ret) > 0 and ret + tok in tok_str:
                    pass
                elif tok == '"':
                    if double_quote_appear:
                        ret = ret + ' '
                elif tok[0] not in alphabet:
                    pass
                elif (ret[-1] not in ['(', '/', u'\u2013', '#', '$', '&']) \
                        and (ret[-1] != '"' or not double_quote_appear):
                    ret = ret + ' '
                ret = ret + tok
            return ret.strip()

        pred_agg, pred_sel, pred_cond = pred_entry
        agg_score, sel_cond_score, cond_op_str_score = score

        sel_num_score, cond_num_score, sel_score, cond_col_score, agg_num_score, agg_op_score, gby_num_score, gby_score, ody_num_score, ody_score, ody_agg_score, ody_par_score = [x.data.cpu().numpy() if x is not None else None for x in sel_cond_score]

        ret_queries = []
        if pred_agg:
            B = len(agg_score)
        elif pred_sel:
            B = len(sel_score)
        elif pred_cond:
            B = len(cond_num_score)
        for b in range(B):
            cur_query = {}
            if pred_sel:
                sel_num_cols = np.argmax(sel_num_score[b]) 
                cur_query['sel_num'] = sel_num_cols
                cur_query['sel'] = np.argsort(-sel_score[b])[:sel_num_cols+1]

                gby_num_cols = np.argmax(gby_num_score[b])
                cur_query['gby_num'] = gby_num_cols
                cur_query['group'] = np.argsort(-gby_score[b])[:gby_num_cols]

                ody_num_cols = np.argmax(ody_num_score[b])
                cur_query['ody_num'] = ody_num_cols
                cur_query['order'] = np.argsort(-ody_score[b])[:ody_num_cols]

                ody_agg_preds = []
                for idx in range(len(gt_ody[b])):           # eventually dont use gold (look at agg query generation)
                    curr_ody_agg = np.argmax(ody_agg_score[b][idx])
                    ody_agg_preds += curr_ody_agg

                cur_query['ody_agg'] = ody_agg_preds
                cur_query['parity'] = np.argmax(ody_par_score[b]) - 1

             
            if pred_agg:
                agg_nums = []
                agg_preds = []
                for idx in range(len(set(gt_sel[b]))):
                # for idx in range(sel_num_cols + 1):
                    curr_num_aggs = np.argmax(agg_num_score[b][idx])
                    agg_nums.append(curr_num_aggs)
                    if curr_num_aggs == 0:
                        curr_agg_ops = [0]
                    else:
                        # curr_agg_ops = list(np.argsort(-agg_op_score[b][idx])[:curr_num_aggs])
                        curr_agg_ops = [x for x in list(np.argsort(-agg_op_score[b][idx])) if x != 0][:curr_num_aggs]
                    agg_preds += curr_agg_ops

                cur_query['agg_num'] = agg_nums
                cur_query['agg'] = agg_preds

                # cur_query['agg'] = np.argmax(agg_score[b].data.cpu().numpy())
            if pred_cond:
                #cond_op_score, cond_str_score = [x.data.cpu().numpy() for x in cond_op_str_score]
                cond_op_score = [x.data.cpu().numpy() for x in cond_op_str_score[0]]
                cur_query['conds'] = []
                cond_num = np.argmax(cond_num_score[b])
                all_toks = ['<BEG>'] + q[b] + ['<END>']
                max_idxes = np.argsort(-cond_col_score[b])[:cond_num]
                for idx in range(cond_num):
                    cur_cond = []
                    cur_cond.append(max_idxes[idx])
                    cur_cond.append(np.argmax(cond_op_score[b][idx]))
                    # cur_cond_str_toks = []
                    # for str_score in cond_str_score[b][idx]:
                    #     str_tok = np.argmax(str_score[:len(all_toks)])
                    #     str_val = all_toks[str_tok]
                    #     if str_val == '<END>':
                    #         break
                    #     #add string word/grouped words to current cond str tokens ["w1", "w2" ...]
                    #     cur_cond_str_toks.append(str_val)
                    # cur_cond.append(merge_tokens(cur_cond_str_toks, raw_q[b]))
                    cur_query['conds'].append(cur_cond)
            ret_queries.append(cur_query)

        return ret_queries
