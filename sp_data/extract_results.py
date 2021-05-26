import numpy as np
import pandas as pd
import os
import pickle
from sklearn.metrics import f1_score
def extract_seq_2_name_dict():
    seq_and_name_df = pd.read_csv("N_sequences.txt", sep='\t')
    name2seq = {}
    for name, seq in zip(seq_and_name_df.ID, seq_and_name_df.DNA_seq):
        name2seq[name.replace("|","_")] = seq
    return name2seq

def extract_relevant_seq_results(test_seqs, labels, results, name2lbl):
    lbls, preds = [], []
    for r in results[2:]:
        name, _, pos_pred, _, _ = r.replace("\n", "").split("\t")
        name = name.replace("_", "|")

        if name in name2lbl:
            lbl = name2lbl[name]
            lbls.append(lbl)
            preds.append(pos_pred)
    return lbls, preds

def get_pos_neg_results(preds,lbls, t):
    lbls = np.array(lbls)
    pos_inds = set(list(np.argwhere(preds > 0.9).reshape(-1)))
    neg_inds = set(list(np.argwhere(preds < 0.1).reshape(-1)))
    actual_pos_inds = set(list(np.argwhere(lbls > 0.9 ).reshape(-1)) )
    actual_neg_inds = set(list(np.argwhere(lbls < 0.1 ).reshape(-1)) )
    pos_acc =len(actual_pos_inds.intersection(pos_inds))/len(actual_pos_inds)
    neg_acc =len(actual_neg_inds.intersection(neg_inds))/len(actual_neg_inds)
    print("Positive/negative accuracies for SP5 model for threshold {} are {}/{}".format(t, pos_acc,neg_acc))

def get_best_acc_by_threshold(all_relevant_lbls, pos_preds):
    pos_preds = [float(p_p) for p_p in pos_preds]
    pos_preds = np.array(pos_preds)
    best_res_threshold, best_res_f1 = 0, 0
    best_preds = np.zeros(len(pos_preds))
    best_pos_preds = list(np.where(pos_preds > 0.5)[0])
    print(len(best_pos_preds), len(all_relevant_lbls))
    best_preds[best_pos_preds] = 1
    print(best_res_f1, best_res_threshold)
    get_pos_neg_results(best_preds, all_relevant_lbls, best_res_threshold)

def extract_sp5_results(aa_len, pos_threshold=0.9, neg_threshold=0.15):
    # compare the sequences that respect the thresholds (p>pos_threshhold or  p<neg_threshold)
    # use for comprehensive
    test_seqs, labels, \
    names  = pickle.load(open("raw_seq_data_{}_{}.bin".format(pos_threshold, neg_threshold), 'rb'))[0],\
             pickle.load(open("raw_seq_data_{}_{}.bin".format(pos_threshold, neg_threshold), 'rb'))[1], \
             pickle.load(open("raw_seq_data_{}_{}.bin".format(pos_threshold, neg_threshold), 'rb'))[2]
    name2lbl = {n:l for n,l in zip(names, labels)}
    files = os.listdir("sp5_outputs/"+str(aa_len) +"aa/")
    all_relevant_seqs, all_relevant_lbls, pos_preds = [], [], []
    for f in files:
        results = open("sp5_outputs/" + str(aa_len) +"aa/" + f,'rt').readlines()
        current_lbls, preds = extract_relevant_seq_results(test_seqs, labels, results, name2lbl)
        all_relevant_lbls.extend(current_lbls)
        # print(f, np.where(np.array([float(p) for p in preds]) > 0.5)[0] )
        pos_preds.extend(preds)
    print(len(pos_preds), len(all_relevant_seqs), len(files))
    get_best_acc_by_threshold(all_relevant_lbls, pos_preds)
    # print(results.head())


def extract_results_for_mdl(mdl):
    result_files = os.listdir("results/")
    current_mdl_results = []
    for f in result_files:
        if mdl in f:
            results_for_fold = pickle.load(open("results/"+f, "rb"))[-1]
            current_mdl_results.extend(results_for_fold)
    test_ds_lens, neg_acc, positive_acc = [r[0] for r in current_mdl_results], \
                                     [r[1] for r in current_mdl_results],\
                                     [r[2] for r in current_mdl_results]
    weights = [t_ds_len/sum(test_ds_lens) for t_ds_len in test_ds_lens]
    pos_summary = sum([positive_acc[i] * weights[i] for i in range(len(weights))])
    neg_summary = sum([neg_acc[i] * weights[i] for i in range(len(weights))])
    print("For model {} the positive/negative accuracies are: {}/{}".format(mdl, pos_summary, neg_summary))

if __name__== "__main__":
    results_folder = "results/"
    result_files = os.listdir(results_folder)
    mdls = set()
    for f in result_files:
        if "results" in f:
            mdls.add(f.split("_results")[0])
    for m in mdls:
        extract_results_for_mdl(m)
    extract_sp5_results(100)