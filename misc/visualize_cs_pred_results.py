from sklearn.metrics import matthews_corrcoef as  compute_mcc
import os
import pickle
from Bio import SeqIO
import numpy as np


def get_cs_acc(life_grp, seqs, true_lbls, pred_lbls, v=False):
    def is_cs(predicted_cs_ind, true_lbl_seq):
        if true_lbl_seq[predicted_cs_ind] == "S" and true_lbl_seq[predicted_cs_ind + 1] != "S":
            return True
        return False

    def get_acc_for_tolerence(ind, t_lbl, v=False):
        true_cs = 0
        while t_lbl[true_cs] == "S" and true_cs < len(t_lbl):
            true_cs += 1
        if np.abs(true_cs - ind) == 0:
            return np.array([1, 1, 1, 1, 1, 0])
        elif np.abs(true_cs - ind) == 1:
            return np.array([0, 1, 1, 1, 1, 0])
        elif np.abs(true_cs - ind) == 2:
            return np.array([0, 0, 1, 1, 1, 0])
        elif np.abs(true_cs - ind) == 3:
            return np.array([0, 0, 0, 1, 1, 0])
        else:
            return np.array([0, 0, 0, 0, 1, 0])

    # S = signal_peptide; T = Tat/SPI or Tat/SPII SP; L = Sec/SPII SP; P = SEC/SPIII SP; I = cytoplasm; M = transmembrane; O = extracellular;
    # order of elemnts in below list:
    # (eukaria_correct_tollerence_0, eukaria_correct_tollerence_1, eukaria_correct_tollerence_2, ..3, eukaria_total, eukaria_all_pos_preds)
    # (negative_correct_tollerence_0, negative_correct_tollerence_1, negative_correct_tollerence_2, ..3, negative_total, negative_all_pos_preds)
    # (positive_correct_tollerence_0, positive_correct_tollerence_1, positive_correct_tollerence_2, ..3, positive_total, positive_all_pos_preds)
    # (archaea_correct_tollerence_0, archaea_correct_tollerence_1, archaea_correct_tollerence_2, ..3, archaea_total, archae_all_pos_preds)
    # We used precision and recall to assess CS predictions, where precision is defined as the fraction of CS predictions
    # that are correct, and recall is the fraction of real SPs that are predicted as the correct SP type and have the correct CS assigned.
    grp2_ind = {"EUKARYA": 0, "NEGATIVE": 1, "POSITIVE": 2, "ARCHAEA": 3}
    predictions = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
    predictions = [np.array(p) for p in predictions]
    count, count2 = 0, 0
    for l, s, t, p in zip(life_grp, seqs, true_lbls, pred_lbls):
        life_grp, sp_info = l.split("|")
        ind = 0
        if sp_info == "SP":
            count+=1
            # print(p,)
            # print(t)
            # print(s)
            # print("\n")
            while p[ind] == "S" and ind < len(p) - 1:
                ind += 1
            predictions[grp2_ind[life_grp]] += get_acc_for_tolerence(ind, t, v=p)
        elif sp_info != "SP" and p[ind] == "S" and l.split("|")[0] == "EUKARYA":
            # count false positive predictions
            predictions[grp2_ind[life_grp]] += np.array([0, 0, 0, 0, 0, 1])
    all_recalls = []
    all_precisions = []
    total_positives = []
    false_positives = []
    for life_grp, ind in grp2_ind.items():
        current_preds = predictions[grp2_ind[life_grp]]
        if v:
            print("{}: {}".format(life_grp, [current_preds[i] / current_preds[-2] for i in range(len(current_preds) - 2)]))
            print("{}: {}".format(life_grp, [current_preds[i] / current_preds[-1] for i in range(len(current_preds) - 2)]))
        all_recalls.append([current_preds[i]/current_preds[-2] for i in range(len(current_preds) -2)])
        all_precisions.append([])
        for i in range(4):
            if current_preds[-1] + current_preds[i] == 0:
                all_precisions[-1].append(0.)
            else:
                all_precisions[-1].append(current_preds[i]/(current_preds[-1] + current_preds[i]))
        total_positives.append(current_preds[-2])
        false_positives.append(current_preds[-1])
    return all_recalls, all_precisions, total_positives, false_positives, predictions

def get_pred_accs_sp_vs_nosp(life_grp, seqs, true_lbls, pred_lbls, v=False):
    # S = signal_peptide; T = Tat/SPI or Tat/SPII SP; L = Sec/SPII SP; P = SEC/SPIII SP; I = cytoplasm; M = transmembrane; O = extracellular;
    # order of elemnts in below list:
    # (eukaria_tp, eukaria_tn, eukaria_fp, eukaria_fn)
    # (negative_tp, negative_tn, negative_fp, negative_fn)
    # (positive_tp, positive_tn, positive_fp, positive_fn)
    # (archaea_correct, archaea_total)
    # Matthews correlation coefficient (MCC) both true and false positive and negative predictions are counted at
    # the sequence level
    grp2_ind = {"EUKARYA": 0, "NEGATIVE": 1, "POSITIVE": 2, "ARCHAEA": 3}
    predictions = [[[], []], [[], []], [[], []], [[], []]]
    for l, s, t, p in zip(life_grp, seqs, true_lbls, pred_lbls):
        life_grp, sp_info = l.split("|")
        ind = 0
        life_grp, sp_info = l.split("|")
        if sp_info == "SP" or sp_info == "NO_SP":
            p = p.replace("ES", "J")
            len_ = min(len(p), len(t))
            t, p = t[:len_], p[:len_]
            for ind in range(len(t)):
                if t[ind] == "S" and p[ind] == "S":
                    predictions[grp2_ind[life_grp]][1].append(1)
                    predictions[grp2_ind[life_grp]][0].append(1)
                elif t[ind] == "S" and p[ind] != "S":
                    predictions[grp2_ind[life_grp]][1].append(1)
                    predictions[grp2_ind[life_grp]][0].append(-1)
                elif t[ind] != "S" and p[ind] == "S":
                    predictions[grp2_ind[life_grp]][1].append(-1)
                    predictions[grp2_ind[life_grp]][0].append(1)
                elif t[ind] != "S" and p[ind] != "S":
                    predictions[grp2_ind[life_grp]][1].append(-1)
                    predictions[grp2_ind[life_grp]][0].append(-1)
    mccs = []
    for grp, id in grp2_ind.items():
        if sum(predictions[grp2_ind[grp]][0]) == -len(predictions[grp2_ind[grp]][0]) or \
                sum(predictions[grp2_ind[grp]][0]) == len(predictions[grp2_ind[grp]][0]):
            mccs.append(-1)
        else:
            mccs.append(compute_mcc(predictions[grp2_ind[grp]][0]
                                                ,predictions[grp2_ind[grp]][1]))
        if v:
            print("{}: {}".format(grp, mccs[-1] ))

    return mccs


def extract_seq_group_for_predicted_aa_lbls(filename="run_wo_lg_info.bin", test_fold=2):
    seq2preds = pickle.load(open(filename, "rb"))
    tested_seqs = set(seq2preds.keys())
    seq2id = {}
    life_grp, seqs, true_lbls, pred_lbls = [], [], [], []
    for seq_record in SeqIO.parse(get_data_folder() + "sp6_data/train_set.fasta", "fasta"):
        seq, lbls = seq_record.seq[:len(seq_record.seq) // 2], seq_record.seq[len(seq_record.seq) // 2:]
        if seq in tested_seqs:
            life_grp.append("|".join(str(seq_record.id).split("|")[1:-1]))
            seqs.append(seq)
            true_lbls.append(lbls)
            pred_lbls.append(seq2preds[seq])
    return life_grp, seqs, true_lbls, pred_lbls


def get_data_folder():
    if os.path.exists("/scratch/work/dumitra1"):
        return "/scratch/work/dumitra1/sp_data/"
    elif os.path.exists("/home/alex"):
        return "sp_data/"
    else:
        return "/scratch/project2003818/dumitra1/sp_data/"


def get_cs_and_sp_pred_results(filename="run_wo_lg_info.bin", v=False):
    life_grp, seqs, true_lbls, pred_lbls = extract_seq_group_for_predicted_aa_lbls(filename=filename)
    sp_pred_accs = get_pred_accs_sp_vs_nosp(life_grp, seqs, true_lbls, pred_lbls, v=v)
    all_recalls, all_precisions, total_positives, false_positives, predictions = get_cs_acc(life_grp, seqs, true_lbls, pred_lbls, v=v)
    return sp_pred_accs, all_recalls, all_precisions, total_positives, false_positives, predictions

def get_summary_sp_acc(sp_pred_accs):
    return np.mean(sp_pred_accs), sp_pred_accs[0]

def get_summary_cs_acc(all_cs_preds):
    return np.mean(np.array(all_cs_preds)), np.mean(all_cs_preds[0]), all_cs_preds[0][0]

if __name__ == "__main__":
    life_grp, seqs, true_lbls, pred_lbls = extract_seq_group_for_predicted_aa_lbls(filename="w_lg_w_glbl_lbl_100ep.bin")
    sp_pred_accs = get_pred_accs_sp_vs_nosp(life_grp, seqs, true_lbls, pred_lbls,v=True)
    all_recalls, all_precisions, total_positives, false_positives, predictions = get_cs_acc(life_grp, seqs, true_lbls, pred_lbls)

