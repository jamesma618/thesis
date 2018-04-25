import json
import torch
import datetime
import argparse
import numpy as np
from scripts.utils import *
from scripts.model.sqlnet import SQLNet

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--toy', action='store_true',
            help='If set, use small data; used for fast debugging.')
    parser.add_argument('--suffix', type=str, default='',
            help='The suffix at the end of saved model name.')
    parser.add_argument('--sd', type=str, default='',
            help='set model save directory.')
    parser.add_argument('--dataset', type=int, default=0,
            help='0: original dataset, 1: re-split dataset, 2: new complex dataset')
    parser.add_argument('--train_emb', action='store_true',
            help='Train word embedding.')

    args = parser.parse_args()

    N_word=300
    B_word=42
    if args.toy:
        USE_SMALL=True
        GPU=True
        BATCH_SIZE=20
    else:
        USE_SMALL=False
        GPU=True
        BATCH_SIZE=64
    TRAIN_ENTRY=(True, True, True)  # (AGG, SEL, COND)
    TRAIN_AGG, TRAIN_SEL, TRAIN_COND = TRAIN_ENTRY
    learning_rate = 1e-3

    sql_data, table_data, val_sql_data, val_table_data, \
            test_sql_data, test_table_data, \
            TRAIN_DB, DEV_DB, TEST_DB = load_dataset(args.dataset, use_small=USE_SMALL)

    # sql_data = np.random.choice(sql_data, 10, replace=False)
    # val_sql_data = test_sql_data = sql_data

    # sql_data = sql_data[:5]
    # val_sql_data = val_sql_data[:5]
    # test_sql_data = test_sql_data[:5]

    # for i, item in enumerate(sql_data):
    #     print 'Question ' + str(i)
    #     print 'Question:', item['question']
    #     print 'Gold Query:', item['query']
    #     print 'SQL:', item['sql1']
    #     print '\n'

    # sys.sleep()


    word_emb = load_word_emb('../alt/glove/glove.%dB.%dd.txt'%(B_word,N_word), \
            load_used=args.train_emb, use_small=USE_SMALL)
    #word_emb = load_concat_wemb('glove/glove.42B.300d.txt', "/data/projects/paraphrase/generation/para-nmt-50m/data/paragram_sl999_czeng.txt")

    model = SQLNet(word_emb, N_word=N_word, gpu=GPU, trainable_emb=args.train_emb)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay = 0)

    agg_m, sel_m, cond_m = best_model_name(args)

    if args.train_emb: # Load pretrained model.
        agg_lm, sel_lm, cond_lm = best_model_name(args, for_load=True)
        print "Loading from %s"%agg_lm
        model.agg_pred.load_state_dict(torch.load(agg_lm))
        print "Loading from %s"%sel_lm
        model.selcond_pred.load_state_dict(torch.load(sel_lm))
        print "Loading from %s"%cond_lm
        model.cond_pred.load_state_dict(torch.load(cond_lm))


    #initial accuracy
    init_acc = epoch_acc(model, BATCH_SIZE, val_sql_data, val_table_data, TRAIN_ENTRY)
    best_agg_acc = init_acc[1][1]
    best_agg_idx = 0
    best_sel_acc = init_acc[1][2]
    best_sel_idx = 0
    best_cond_acc = init_acc[1][3]
    best_cond_idx = 0
    print 'Init dev acc_qm: %s\n  breakdown results: %s' % init_acc
    if TRAIN_AGG:
        torch.save(model.agg_pred.state_dict(), agg_m)
    if TRAIN_SEL:
        torch.save(model.selcond_pred.state_dict(), sel_m)
    if TRAIN_COND:
        torch.save(model.op_str_pred.state_dict(), cond_m)

    for i in range(300):
        print 'Epoch %d @ %s'%(i+1, datetime.datetime.now())
        print ' Loss = %s'%epoch_train(
                model, optimizer, BATCH_SIZE,
                sql_data, table_data, TRAIN_ENTRY)
        train_tot_acc, train_bkd_acc = epoch_acc(model, BATCH_SIZE, sql_data, table_data, TRAIN_ENTRY, train_flag = True)
        print ' Train acc_qm: %s' % train_tot_acc
        print ' Breakdown results: agg #: %s, agg: %s, sel: %s, cond: %s, sel #: %s, cond #: %s, cond col: %s, cond op: %s, cond val: %s, group #: %s, group: %s, order #: %s, order: %s, order agg: %s, order par: %s'\
            % (train_bkd_acc[0], train_bkd_acc[1], train_bkd_acc[2], train_bkd_acc[3], train_bkd_acc[4], train_bkd_acc[5], train_bkd_acc[6], train_bkd_acc[7], train_bkd_acc[8], train_bkd_acc[9], train_bkd_acc[10], train_bkd_acc[11], train_bkd_acc[12], train_bkd_acc[13], train_bkd_acc[14])

        val_tot_acc, val_bkd_acc = epoch_acc(model, BATCH_SIZE, val_sql_data, val_table_data, TRAIN_ENTRY, error_print = False, train_flag = False) #for detailed error analysis, pass True to error_print
        print ' Dev acc_qm: %s' % val_tot_acc
        print ' Breakdown results: agg #: %s, agg: %s,  sel: %s, cond: %s, sel #: %s, cond #: %s, cond col: %s, cond op: %s, cond val: %s, group #: %s, group: %s, order #: %s, order: %s, order agg: %s, order par: %s'\
            % (val_bkd_acc[0], val_bkd_acc[1], val_bkd_acc[2], val_bkd_acc[3], val_bkd_acc[4], val_bkd_acc[5], val_bkd_acc[6], val_bkd_acc[7], val_bkd_acc[8], val_bkd_acc[9], val_bkd_acc[10], val_bkd_acc[11], val_bkd_acc[12], val_bkd_acc[13], val_bkd_acc[14])

        if TRAIN_AGG:
            if val_bkd_acc[1] > best_agg_acc:
                best_agg_acc = val_bkd_acc[1]
                best_agg_idx = i+1
                torch.save(model.agg_pred.state_dict(),
                    args.sd + '/epoch%d.agg_model%s'%(i+1, args.suffix))
                torch.save(model.agg_pred.state_dict(), agg_m)

        if TRAIN_SEL:
            if val_bkd_acc[2] > best_sel_acc:
                best_sel_acc = val_bkd_acc[2]
                best_sel_idx = i+1
                torch.save(model.selcond_pred.state_dict(),
                    args.sd + '/epoch%d.sel_model%s'%(i+1, args.suffix))
                torch.save(model.selcond_pred.state_dict(), sel_m)

        if TRAIN_COND:
            if val_bkd_acc[3] > best_cond_acc:
                best_cond_acc = val_bkd_acc[3]
                best_cond_idx = i+1
                torch.save(model.op_str_pred.state_dict(),
                    args.sd + '/epoch%d.cond_model%s'%(i+1, args.suffix))
                torch.save(model.op_str_pred.state_dict(), cond_m)

        print ' Best val acc = %s, on epoch %s individually'%(
                (best_agg_acc, best_sel_acc, best_cond_acc),
                (best_agg_idx, best_sel_idx, best_cond_idx))
