import torch.nn
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef as compute_mcc
import os
import pickle
from Bio import SeqIO
import numpy as np


def get_cs_acc(life_grp, seqs, true_lbls, pred_lbls, v=False, only_cs_position=False, sp_type="SP", sptype_preds=None):
    def get_acc_for_tolerence(ind, t_lbl, sp_letter):
        true_cs = 0
        while t_lbl[true_cs] == sp_letter and true_cs < len(t_lbl):
            true_cs += 1
        if np.abs(true_cs - ind) == 0:
            return np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        elif np.abs(true_cs - ind) == 1:
            return np.array([0, 1, 1, 1, 1, 0, 1, 0, 0, 0])
        elif np.abs(true_cs - ind) == 2:
            return np.array([0, 0, 1, 1, 1, 0, 1, 1, 0, 0])
        elif np.abs(true_cs - ind) == 3:
            return np.array([0, 0, 0, 1, 1, 0, 1, 1, 1, 0])
        elif ind != 0:
            # if ind==0, SP was predicted, but CS prediction is off for all tolerence levels, meaning it's a false positive
            # if ind != 0, and teh corresponding SP was predicted, this becomes a false positive on CS predictions for all
            # tolerance levels
            return np.array([0, 0, 0, 0, 1, 0, 1, 1, 1, 1])
        else:
            # if ind==0, SP was not even predicted (so there is no CS prediction) and this affects precision metric
            # tp/(tp+fp). It means this a fn
            return np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0])

    ind2glbl_lbl = {0: 'NO_SP', 1: 'SP', 2: 'TATLIPO', 3: 'LIPO', 4: 'TAT', 5: 'PILIN'}
    sptype2letter = {'TAT': 'T', 'LIPO': 'L', 'PILIN': 'P', 'TATLIPO': 'T', 'SP': 'S'}
    sp_types = ["S", "T", "L", "P"]
    # S = signal_peptide; T = Tat/SPI or Tat/SPII SP; L = Sec/SPII SP; P = SEC/SPIII SP; I = cytoplasm; M = transmembrane; O = extracellular;
    # order of elemnts in below list:
    # (eukaria_correct_tollerence_0, eukaria_correct_tollerence_1, eukaria_correct_tollerence_2, ..3, eukaria_total, eukaria_all_pos_preds)
    # (negative_correct_tollerence_0, negative_correct_tollerence_1, negative_correct_tollerence_2, ..3, negative_total, negative_all_pos_preds)
    # (positive_correct_tollerence_0, positive_correct_tollerence_1, positive_correct_tollerence_2, ..3, positive_total, positive_all_pos_preds)
    # (archaea_correct_tollerence_0, archaea_correct_tollerence_1, archaea_correct_tollerence_2, ..3, archaea_total, archae_all_pos_preds)
    # We used precision and recall to assess CS predictions, where precision is defined as the fraction of CS predictions
    # that are correct, and recall is the fraction of real SPs that are predicted as the correct SP type and have the correct CS assigned.
    grp2_ind = {"EUKARYA": 0, "NEGATIVE": 1, "POSITIVE": 2, "ARCHAEA": 3}
    predictions = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    predictions = [np.array(p) for p in predictions]
    count, count2 = 0, 0
    count_tol_fn, count_complete_fn, count_otherSPpred = 0, 0, 0
    sp_letter = sptype2letter[sp_type]
    for l, s, t, p in zip(life_grp, seqs, true_lbls, pred_lbls):
        lg, sp_info = l.split("|")
        ind = 0
        predicted_sp = p[0]
        is_sp = predicted_sp in sp_types
        if sp_info == sp_type:

            #     # the precision was defined as the fraction of correct CS predictions over the number of predicted
            #     # CS, recall as the fraction of correct CS predictions over the number of true CS. In both cases,
            #     # a CS was only considered correct if it was predicted in the correct SP class (e.g. when the
            #     # model predicts a CS in a Sec/SPI sequence, but predicts Sec/SPII as the sequence label, the
            #     # sample is considered as no CS predicted).

            # SO ONLY CONSIDER IT AS A FALSE NEGATIVE FOR THE APPROPRIATE correct-SP class?
            if (sptype_preds is not None and ind2glbl_lbl[sptype_preds[s]] == sp_type) or (sptype_preds is None and sp_letter == p[ind]):
                while (p[ind] == sp_letter or (p[ind] == predicted_sp and is_sp and only_cs_position)) and ind < len(p) - 1:
                    # when only_cs_position=True, the cleavage site positions will be taken into account irrespective of
                    # whether the predicted SP is the correct kind
                    ind += 1
            else:
                ind = 0
            predictions[grp2_ind[lg]] += get_acc_for_tolerence(ind, t, sp_letter)
        # elif sp_info != sp_type and   p[ind] == sp_letter:
        elif (sptype_preds is not None and sp_info != sp_type and ind2glbl_lbl[sptype_preds[s]] == sp_type) or (sptype_preds is None and p[ind] == sp_letter):
            predictions[grp2_ind[lg]] += np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
    if v:
        print(" count_tol_fn, count_complete_fn, count_otherSPpred", count_tol_fn, count_complete_fn, count_otherSPpred)
    all_recalls = []
    all_precisions = []
    all_f1_scores = []
    total_positives = []
    false_positives = []
    for life_grp, ind in grp2_ind.items():
        if sp_type == "SP" or life_grp != "EUKARYA":
            # eukaryotes do not have SEC/SPI or SEC/SPII
            current_preds = predictions[grp2_ind[life_grp]]
            if v:
                print("Recall {}: {}".format(life_grp, [current_preds[i] / current_preds[4] for i in range(4)]))
                print("Prec {}: {}".format(life_grp,
                                           [current_preds[i] / (current_preds[i] + current_preds[5]) for i in
                                            range(4)]))
            all_recalls.append([current_preds[i] / current_preds[4] for i in range(4)])
            all_precisions.append([])
            all_f1_scores.append([])
            for i in range(4):
                if current_preds[5] + current_preds[i] == 0:
                    all_precisions[-1].append(0.)
                else:
                    all_precisions[-1].append(
                        current_preds[i] / (current_preds[i] + current_preds[i + 6]))
            current_recs, current_precs = all_recalls[-1], all_precisions[-1]
            all_f1_scores[-1].extend([0 if current_recs[i] * current_precs[i] == 0 else 2 * current_recs[i] *
                                                                                        current_precs[i] / (
                                                                                                    current_recs[i] +
                                                                                                    current_precs[i])
                                      for i in range(4)])
            total_positives.append(current_preds[4])
            false_positives.append(current_preds[5])
    return all_recalls, all_precisions, total_positives, false_positives, predictions, all_f1_scores


def get_class_sp_accs(life_grp, seqs, true_lbls, pred_lbls):
    groups_tp_tn_fp_fn = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    grp2_ind = {"EUKARYA": 0, "NEGATIVE": 1, "POSITIVE": 2, "ARCHAEA": 3}
    for lg, s, tl, pl in zip(life_grp, seqs, true_lbls, pred_lbls):
        lg = lg.split("|")[0]
        if tl[0] == pl[0] and tl[0] == "S":
            groups_tp_tn_fp_fn[grp2_ind[lg]][0] += 1
        elif tl[0] != "S" and pl[0] != "S":
            groups_tp_tn_fp_fn[grp2_ind[lg]][1] += 1
        elif tl[0] == "S" and pl[0] != "S":
            groups_tp_tn_fp_fn[grp2_ind[lg]][3] += 1
        elif tl[0] != "S" and pl[0] == "S":
            groups_tp_tn_fp_fn[grp2_ind[lg]][2] += 1
    recs = [groups_tp_tn_fp_fn[i][0] / (groups_tp_tn_fp_fn[i][0] + groups_tp_tn_fp_fn[i][3]) if
            groups_tp_tn_fp_fn[i][0] + groups_tp_tn_fp_fn[i][3] != 0 else 0 for i in range(4)]
    precs = [groups_tp_tn_fp_fn[i][0] / (groups_tp_tn_fp_fn[i][0] + groups_tp_tn_fp_fn[i][2]) if
             groups_tp_tn_fp_fn[i][0] + groups_tp_tn_fp_fn[i][2] != 0 else 0 for i in range(4)]
    return [ (2 * recs[i] * precs[i]) / (precs[i] + recs[i]) if precs[i] + recs[i] != 0 else 0 for i in range(4)]


def get_pred_accs_sp_vs_nosp(life_grp, seqs, true_lbls, pred_lbls, v=False, return_mcc2=False, sp_type="SP", sptype_preds=None):
    # S = signal_peptide; T = Tat/SPI or Tat/SPII SP; L = Sec/SPII SP; P = SEC/SPIII SP; I = cytoplasm; M = transmembrane; O = extracellular;
    # order of elemnts in below list:
    # (eukaria_tp, eukaria_tn, eukaria_fp, eukaria_fn)
    # (negative_tp, negative_tn, negative_fp, negative_fn)
    # (positive_tp, positive_tn, positive_fp, positive_fn)
    # (archaea_correct, archaea_total)
    # Matthews correlation coefficient (MCC) both true and false positive and negative predictions are counted at
    # the sequence level
    grp2_ind = {"EUKARYA": 0, "NEGATIVE": 1, "POSITIVE": 2, "ARCHAEA": 3}
    lg2sp_letter = {'TAT': 'T', 'LIPO': 'L', 'PILIN': 'P', 'TATLIPO': 'T', 'SP': 'S'}
    sp_type_letter = lg2sp_letter[sp_type]
    predictions = [[[], []], [[], []], [[], []], [[], []]]
    predictions_mcc2 = [[[], []], [[], []], [[], []], [[], []]]
    zv = 0
    for l, s, t, p in zip(life_grp, seqs, true_lbls, pred_lbls):
        zv += 1
        lg, sp_info = l.split("|")
        if sp_info == sp_type or sp_info == "NO_SP":
            p = p.replace("ES", "J")
            len_ = min(len(p), len(t))
            t, p = t[:len_], p[:len_]
            for ind in range(len(t)):
                if t[ind] == sp_type_letter and p[ind] == sp_type_letter:
                    predictions[grp2_ind[lg]][1].append(1)
                    predictions[grp2_ind[lg]][0].append(1)
                elif t[ind] == sp_type_letter and p[ind] != sp_type_letter:
                    predictions[grp2_ind[lg]][1].append(1)
                    predictions[grp2_ind[lg]][0].append(-1)
                elif t[ind] != sp_type_letter and p[ind] == sp_type_letter:
                    predictions[grp2_ind[lg]][1].append(-1)
                    predictions[grp2_ind[lg]][0].append(1)
                elif t[ind] != sp_type_letter and p[ind] != sp_type_letter:
                    predictions[grp2_ind[lg]][1].append(-1)
                    predictions[grp2_ind[lg]][0].append(-1)
        if return_mcc2:
            p = p.replace("ES", "J")
            len_ = min(len(p), len(t))
            t, p = t[:len_], p[:len_]
            for ind in range(len(t)):
                if t[ind] == sp_type_letter and p[ind] == sp_type_letter:
                    predictions_mcc2[grp2_ind[lg]][1].append(1)
                    predictions_mcc2[grp2_ind[lg]][0].append(1)
                elif t[ind] == sp_type_letter and p[ind] != sp_type_letter:
                    predictions_mcc2[grp2_ind[lg]][1].append(1)
                    predictions_mcc2[grp2_ind[lg]][0].append(-1)
                elif t[ind] != sp_type_letter and p[ind] == sp_type_letter:
                    predictions_mcc2[grp2_ind[lg]][1].append(-1)
                    predictions_mcc2[grp2_ind[lg]][0].append(1)
                elif t[ind] != sp_type_letter and p[ind] != sp_type_letter:
                    predictions_mcc2[grp2_ind[lg]][1].append(-1)
                    predictions_mcc2[grp2_ind[lg]][0].append(-1)
    mccs, mccs2 = [], []
    for grp, id in grp2_ind.items():
        if sp_type == "SP" or grp != "EUKARYA":
            if sum(predictions[grp2_ind[grp]][0]) == -len(predictions[grp2_ind[grp]][0]) or \
                    sum(predictions[grp2_ind[grp]][0]) == len(predictions[grp2_ind[grp]][0]):
                mccs.append(-1)
            else:
                mccs.append(compute_mcc(predictions[grp2_ind[grp]][0]
                                        , predictions[grp2_ind[grp]][1]))
            if v:
                print("{}: {}".format(grp, mccs[-1]))
    if return_mcc2:
        for grp, id in grp2_ind.items():
            if sp_type == "SP" or grp != "EUKARYA":

                if sum(predictions_mcc2[grp2_ind[grp]][0]) == -len(predictions_mcc2[grp2_ind[grp]][0]) or \
                        sum(predictions_mcc2[grp2_ind[grp]][0]) == len(predictions_mcc2[grp2_ind[grp]][0]):
                    mccs2.append(-1)
                else:
                    mccs2.append(compute_mcc(predictions_mcc2[grp2_ind[grp]][0]
                                             , predictions_mcc2[grp2_ind[grp]][1]))
                if v:
                    print("{}: {}".format(grp, mccs2[-1]))
        return mccs, mccs2
    return mccs


def get_bin(p, bins):
    for i in range(len(bins)):
        if bins[i] < p <= bins[i + 1]:
            return i


def get_cs_preds_by_tol(tl, pl):
    pl = pl.replace("ES", "")
    correct_by_tol = [0, 0, 0, 0]
    for tol in range(4):
        correct_by_tol[tol] = int(tl.rfind("S") - tol <= pl.rfind("S") <= tl.rfind("S") + tol)
    return correct_by_tol


def plot_reliability_diagrams(resulted_perc_by_acc, name, total_counts_per_acc):
    import matplotlib
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(12, 8)
    fig.savefig('test2png.png', dpi=100)

    accs = [acc_to_perc[0] for acc_to_perc in resulted_perc_by_acc]
    total_counts_per_acc = list(total_counts_per_acc)
    percs = [acc_to_perc[1] for acc_to_perc in resulted_perc_by_acc]
    bars_width = accs[0] - accs[1]
    plt.title(name)
    plt.bar(accs, accs, width=bars_width, alpha=0.5, linewidth=2, edgecolor="black", color='blue',
            label='Perfect calibration')
    plt.bar(accs, percs, width=bars_width, alpha=0.5, color='red', label="Model's calibration",
            tick_label=["{}\n{}".format(str(round(accs[i], 2)), str(total_counts_per_acc[i])) for i in
                        range(len(accs))])
    plt.xlabel("Prob/No of preds")
    plt.ylabel("Prob")
    plt.legend()
    plt.show()


def get_prob_calibration_and_plot(probabilities_file="", life_grp=None, seqs=None, true_lbls=None, pred_lbls=None,
                                  bins=15, plot=True, sp2probs=None):
    # initialize bins
    bin_limmits = np.linspace(0, 1, bins)
    correct_calibration_accuracies = [(bin_limmits[i] + bin_limmits[i + 1]) / 2 for i in range(bins - 1)]

    # initialize cleavage site and binary sp dictionaries of the form:
    # binary_sp_dict: {<life_group> : { total : { <accuracy> : count }, correct : { <accuracy> : count} } }
    # cs_pred: {<life_group> : { <tol> : { total : {<accuracy> : count}, correct: {<accuracy> : count} } } }
    binary_sp_calibration_by_grp = {}
    cs_by_lg_and_tol_accs = {}
    for lg in life_grp:
        lg = lg.split("|")[0]
        crct_cal_acc_2_correct_preds = {crct_cal_acc: 0 for crct_cal_acc in correct_calibration_accuracies}
        crct_cal_acc_2_totals = {crct_cal_acc: 0 for crct_cal_acc in correct_calibration_accuracies}
        wrap_dict = {'total': crct_cal_acc_2_totals, 'correct': crct_cal_acc_2_correct_preds}
        binary_sp_calibration_by_grp[lg] = wrap_dict
        tol_based_cs_accs = {}
        for tol in range(4):
            crct_cal_acc_2_correct_preds = {crct_cal_acc: 0 for crct_cal_acc in correct_calibration_accuracies}
            crct_cal_acc_2_totals = {crct_cal_acc: 0 for crct_cal_acc in correct_calibration_accuracies}
            wrap_dict = {'total': crct_cal_acc_2_totals, 'correct': crct_cal_acc_2_correct_preds}
            tol_based_cs_accs[tol] = wrap_dict
        cs_by_lg_and_tol_accs[lg] = tol_based_cs_accs

    sp2probs = pickle.load(open(probabilities_file, "rb")) if sp2probs is None else sp2probs
    for s, tl, pl, lg in zip(seqs, true_lbls, pred_lbls, life_grp):
        lg = lg.split("|")[0]
        predicted_sp_prob, all_sp_probs, _ = sp2probs[s]
        bin = get_bin(predicted_sp_prob, bin_limmits)
        coresp_acc = correct_calibration_accuracies[bin]
        if tl[0] == "S":
            binary_sp_calibration_by_grp[lg]['total'][coresp_acc] += 1
            if pl[0] == "S":
                binary_sp_calibration_by_grp[lg]['correct'][coresp_acc] += 1
        if tl[0] == pl[0] == "S":
            correct_preds_by_tol = get_cs_preds_by_tol(tl, pl)
            for tol in range(4):
                cs_by_lg_and_tol_accs[lg][tol]['correct'][coresp_acc] += correct_preds_by_tol[tol]
                cs_by_lg_and_tol_accs[lg][tol]['total'][coresp_acc] += 1
    binary_ece, cs_ece = [], [[], [], [], []]
    for lg_ind, lg in enumerate(['EUKARYA', 'NEGATIVE', 'POSITIVE', 'ARCHAEA']):
        correct_binary_preds, total_binary_preds = binary_sp_calibration_by_grp[lg]['correct'].values(), \
                                                   binary_sp_calibration_by_grp[lg]['total'].values()
        results = []
        current_binary_ece = []
        for ind, (crct, ttl) in enumerate(zip(correct_binary_preds, total_binary_preds)):
            actual_acc = crct / ttl if ttl != 0 else 0
            results.append((correct_calibration_accuracies[ind], actual_acc if ttl != 0 else 0))
            current_binary_ece.append(
                np.abs(correct_calibration_accuracies[ind] - actual_acc) * (ttl / sum(total_binary_preds)))
        binary_ece.append(round(sum(current_binary_ece), 3))
        if plot:
            print("Binary preds for {} with ECE {}: ".format(lg, sum(current_binary_ece)), results, total_binary_preds)
            plot_reliability_diagrams(results, "Binary sp pred results for {} with ECE {}".format(lg, round(
                sum(current_binary_ece), 3)), total_binary_preds)
        for tol in range(4):
            correct_cs_preds, total_cs_preds = cs_by_lg_and_tol_accs[lg][tol]['correct'].values(), \
                                               cs_by_lg_and_tol_accs[lg][tol]['total'].values()
            results = []
            current_cs_ece = []
            for ind, (crct, ttl) in enumerate(zip(correct_cs_preds, total_cs_preds)):
                results.append((correct_calibration_accuracies[ind], crct / ttl if ttl != 0 else 0))
                actual_acc = crct / ttl if ttl != 0 else 0
                current_cs_ece.append(
                    np.abs(correct_calibration_accuracies[ind] - actual_acc) * (ttl / sum(total_binary_preds)))
            cs_ece[lg_ind].append(round(sum(current_cs_ece), 3))
            if plot:
                plot_reliability_diagrams(results, "CS pred results for tol {} for {} with ECE {}".format(tol, lg,
                                                                                                          round(
                                                                                                              sum(current_cs_ece),
                                                                                                              3)),
                                          total_cs_preds)
                print("Cs preds for {} for tol {}:".format(lg, tol), results)


def extract_seq_group_for_predicted_aa_lbls(filename="run_wo_lg_info.bin", test_fold=2, dict_=None):
    seq2preds = pickle.load(open(filename, "rb")) if dict_ is None else dict_
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
    if os.path.exists("results"):
        return "../sp_data/"
    elif os.path.exists("/scratch/work/dumitra1"):
        return "/scratch/work/dumitra1/sp_data/"
    elif os.path.exists("/home/alex"):
        return "sp_data/"
    else:
        return "/scratch/project2003818/dumitra1/sp_data/"


def get_cs_and_sp_pred_results(filename="run_wo_lg_info.bin", v=False, probabilities_file=None, return_everything=False,
                               return_class_prec_rec=False,):
    sptype_filename = filename.replace(".bin", "")  + "_sptype.bin"
    if os.path.exists(sptype_filename):
        sptype_preds = pickle.load(open(sptype_filename, "rb"))
    else:
        sptype_preds = None
    life_grp, seqs, true_lbls, pred_lbls = extract_seq_group_for_predicted_aa_lbls(filename=filename)
    if probabilities_file is not None:
        get_prob_calibration_and_plot(probabilities_file, life_grp, seqs, true_lbls, pred_lbls)
    sp_pred_mccs = get_pred_accs_sp_vs_nosp(life_grp, seqs, true_lbls, pred_lbls, v=v)
    all_recalls, all_precisions, total_positives, \
    false_positives, predictions, all_f1_scores = get_cs_acc(life_grp, seqs, true_lbls, pred_lbls, v=v, sptype_preds=sptype_preds)
    if return_everything:
        sp_pred_mccs, sp_pred_mccs2 = get_pred_accs_sp_vs_nosp(life_grp, seqs, true_lbls, pred_lbls, v=v,
                                                               return_mcc2=True, sp_type="SP", sptype_preds=sptype_preds)
        lipo_pred_mccs, lipo_pred_mccs2 = get_pred_accs_sp_vs_nosp(life_grp, seqs, true_lbls, pred_lbls, v=v,
                                                                   return_mcc2=True, sp_type="LIPO", sptype_preds=sptype_preds)
        tat_pred_mccs, tat_pred_mccs2 = get_pred_accs_sp_vs_nosp(life_grp, seqs, true_lbls, pred_lbls, v=v,
                                                                 return_mcc2=True, sp_type="TAT", sptype_preds=sptype_preds)

        all_recalls_lipo, all_precisions_lipo, _, _, _, all_f1_scores_lipo = get_cs_acc(life_grp, seqs, true_lbls,
                                                                                        pred_lbls, v=False,
                                                                                        only_cs_position=False,
                                                                                        sp_type="LIPO", sptype_preds=sptype_preds)
        all_recalls_tat, all_precisions_tat, _, _, _, all_f1_scores_tat = get_cs_acc(life_grp, seqs, true_lbls,
                                                                                     pred_lbls, v=False,
                                                                                     only_cs_position=False,
                                                                                     sp_type="TAT", sptype_preds=sptype_preds)
        if return_class_prec_rec:
            return sp_pred_mccs, sp_pred_mccs2, lipo_pred_mccs, lipo_pred_mccs2, tat_pred_mccs, tat_pred_mccs2, \
                   all_recalls_lipo, all_precisions_lipo, all_recalls_tat, all_precisions_tat, all_f1_scores_lipo, all_f1_scores_tat, \
                   all_recalls, all_precisions, total_positives, false_positives, predictions, all_f1_scores, \
                   get_class_sp_accs(life_grp, seqs, true_lbls, pred_lbls)
        return sp_pred_mccs, sp_pred_mccs2, lipo_pred_mccs, lipo_pred_mccs2, tat_pred_mccs, tat_pred_mccs2, \
               all_recalls_lipo, all_precisions_lipo, all_recalls_tat, all_precisions_tat, all_f1_scores_lipo, all_f1_scores_tat, \
               all_recalls, all_precisions, total_positives, false_positives, predictions, all_f1_scores
    if return_class_prec_rec:
        return sp_pred_mccs, all_recalls, all_precisions, total_positives, false_positives, predictions, all_f1_scores, \
               get_class_sp_accs(life_grp, seqs, true_lbls, pred_lbls)
    return sp_pred_mccs, all_recalls, all_precisions, total_positives, false_positives, predictions, all_f1_scores


def get_summary_sp_acc(sp_pred_accs):
    return np.mean(sp_pred_accs), sp_pred_accs[0]


def get_summary_cs_acc(all_cs_preds):
    return np.mean(np.array(all_cs_preds)), np.mean(all_cs_preds[0]), all_cs_preds[0][0]


def plot_losses(losses, name="param_search_0.2_2048_0.0001_"):
    train_loss, valid_loss = losses
    fig, axs = plt.subplots(1, 1, figsize=(12, 8))

    axs.set_title("Train and validation loss over epochs")
    axs.plot(train_loss, label="Train loss")
    axs.plot(valid_loss, label="Validation loss")
    axs.set_xlabel("Epochs")
    axs.set_ylabel("Loss")
    axs.legend()
    axs.set_ylim(0, 0.2)
    # plt.savefig("/home/alex/Desktop/sp6_ds_transformer_nmt_results/" + name + "loss.png")
    plt.show()


def plot_mcc(mccs, name="param_search_0.2_2048_0.0001_"):
    euk_mcc, neg_mcc, pos_mcc, arc_mcc = mccs
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs[0, 0].plot(euk_mcc, label="Eukaryote mcc")
    axs[0, 0].set_ylabel("mcc")
    axs[0, 0].set_ylim(-1.1, 1.1)
    axs[0, 0].legend()

    axs[0, 1].plot(neg_mcc, label="Negative mcc")
    axs[0, 1].set_ylim(-1.1, 1.1)
    axs[0, 1].legend()

    axs[1, 0].plot(pos_mcc, label="Positive mcc")
    axs[1, 0].legend()
    axs[1, 0].set_ylim(-1.1, 1.1)
    axs[1, 0].set_ylabel("mcc")

    axs[1, 0].set_xlabel("epochs")

    axs[1, 1].plot(arc_mcc, label="Archaea mcc")
    axs[1, 1].legend()

    axs[1, 1].set_ylim(-1.1, 1.1)
    axs[1, 1].set_xlabel("epochs")
    # plt.savefig("/home/alex/Desktop/sp6_ds_transformer_nmt_results/{}_{}.png".format(name, "mcc"))
    plt.show()


def extract_and_plot_prec_recall(results, metric="recall", name="param_search_0.2_2048_0.0001_", sp_type_f1=[[]]):
    cs_res_euk, cs_res_neg, cs_res_pos, cs_res_arc = results
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    for i in range(4):
        axs[0, 0].plot(cs_res_euk[i], label="Eukaryote {} tol={}".format(metric, i))
        if i == 0 and metric == "f1"  and len(sp_type_f1[0]) != 0:
            axs[0, 0].plot(sp_type_f1[0], color='black', label="Eukaryote sp type F1")
        axs[0, 0].set_ylabel(metric)
        axs[0, 0].legend()
        axs[0, 0].set_ylim(-0.1, 1.1)

        axs[0, 1].plot(cs_res_neg[i], label="Negative {} tol={}".format(metric, i))
        if i == 0 and metric == "f1" and len(sp_type_f1[0]) != 0:
            axs[0, 1].plot(sp_type_f1[1], color='black' , label="Negative sp type F1")
        axs[0, 1].legend()
        axs[0, 1].set_ylim(-0.1, 1.1)

        axs[1, 0].plot(cs_res_pos[i], label="Positive {} tol={}".format(metric, i))
        if i == 0  and metric == "f1"  and len(sp_type_f1[0]) != 0:
            axs[1, 0].plot(sp_type_f1[2], color='black', label="Positive sp type F1")
        axs[1, 0].legend()
        axs[1, 0].set_xlabel("epochs")
        axs[1, 0].set_ylim(-0.1, 1.1)
        axs[1, 0].set_ylabel(metric)

        axs[1, 1].plot(cs_res_arc[i], label="Archaea {} tol={}".format(metric, i))
        if i == 0  and metric == "f1" and len(sp_type_f1[0]) != 0:
            axs[1, 1].plot(sp_type_f1[3], color='black', label="Archaea sp type F1")
        axs[1, 1].legend()
        axs[1, 1].set_xlabel("epochs")

        axs[1, 1].set_ylim(-0.1, 1.1)

    # plt.savefig("/home/alex/Desktop/sp6_ds_transformer_nmt_results/{}_{}.png".format(name, metric))
    plt.show()


def visualize_validation(run="param_search_0.2_2048_0.0001_", folds=[0, 1], folder=""):
    all_results = []
    euk_mcc, neg_mcc, pos_mcc, arc_mcc, train_loss, valid_loss, cs_recalls_euk, cs_recalls_neg, cs_recalls_pos, \
    cs_recalls_arc, cs_precs_euk, cs_precs_neg, cs_precs_pos, cs_precs_arc, sp_type_f1 = extract_results(run, folds=folds,
                                                                                                folder=folder)
    all_f1 = [[[], [], [], []], [[], [], [], []], [[], [], [], []], [[], [], [], []]]
    for lg_ind, (lg_rec, lg_prec) in enumerate([(cs_recalls_euk, cs_precs_euk), (cs_recalls_neg, cs_precs_neg),
                                                (cs_recalls_pos, cs_precs_pos), (cs_recalls_arc, cs_precs_arc)]):
        for tol in range(4):
            for prec, rec in zip(lg_rec[tol], lg_prec[tol]):
                all_f1[lg_ind][tol].append(2 * prec * rec / (prec + rec) if prec + rec else 0)
    cs_f1_euk, cs_f1_neg, cs_f1_pos, cs_f1_arc = all_f1
    plot_mcc([euk_mcc, neg_mcc, pos_mcc, arc_mcc], name=run)
    plot_losses([train_loss, valid_loss], name=run)
    extract_and_plot_prec_recall([cs_f1_euk, cs_f1_neg, cs_f1_pos, cs_f1_arc], metric="f1", name=run, sp_type_f1=sp_type_f1)
    extract_and_plot_prec_recall([cs_recalls_euk, cs_recalls_neg, cs_recalls_pos, cs_recalls_arc], metric="recall",
                                 name=run)
    extract_and_plot_prec_recall([cs_precs_euk, cs_precs_neg, cs_precs_pos, cs_precs_arc], metric="precision", name=run)
    # extract_and_plot_losses(lines)


def extract_results(run="param_search_0.2_2048_0.0001_", folds=[0, 1], folder='results_param_s_2/'):
    euk_mcc, neg_mcc, pos_mcc, arc_mcc = [], [], [], []
    train_loss, valid_loss = [], []
    cs_recalls_euk, cs_recalls_neg, cs_recalls_pos, cs_recalls_arc = [[], [], [], []], [[], [], [], []], \
                                                                     [[], [], [], []], [[], [], [], []]
    cs_precs_euk, cs_precs_neg, cs_precs_pos, cs_precs_arc = [[], [], [], []], [[], [], [], []], \
                                                             [[], [], [], []], [[], [], [], []]
    class_preds = [[], [], [], []]
    with open(folder + run + "{}_{}.log".format(folds[0], folds[1]), "rt") as f:
        lines = f.readlines()
    for l in lines:
        if "sp_pred mcc" in l and "VALIDATION" in l:
            mccs = l.split(":")[-1].replace(" ", "").split(",")
            euk_mcc.append(float(mccs[0]))
            neg_mcc.append(float(mccs[1]))
            pos_mcc.append(float(mccs[2]))
            arc_mcc.append(float(mccs[3]))
        elif "train/validation" in l:
            train_l, valid_l = l.split(":")[-1].replace(" ", "").split("/")
            valid_l = valid_l.split(",")[0].replace(" ", "")
            train_l, valid_l = float(train_l), float(valid_l)
            train_loss.append(train_l)
            valid_loss.append(valid_l)
        elif "cs recall" in l and "VALIDATION" in l:
            cs_res = l.split(":")[-1].replace(" ", "").split(",")
            cs_res = [float(c_r) for c_r in cs_res]
            cs_recalls_euk[0].append(cs_res[0])
            cs_recalls_euk[1].append(cs_res[1])
            cs_recalls_euk[2].append(cs_res[2])
            cs_recalls_euk[3].append(cs_res[3])

            cs_recalls_neg[0].append(cs_res[4])
            cs_recalls_neg[1].append(cs_res[5])
            cs_recalls_neg[2].append(cs_res[6])
            cs_recalls_neg[3].append(cs_res[7])

            cs_recalls_pos[0].append(cs_res[8])
            cs_recalls_pos[1].append(cs_res[9])
            cs_recalls_pos[2].append(cs_res[10])
            cs_recalls_pos[3].append(cs_res[11])

            cs_recalls_arc[0].append(cs_res[12])
            cs_recalls_arc[1].append(cs_res[13])
            cs_recalls_arc[2].append(cs_res[14])
            cs_recalls_arc[3].append(cs_res[15])

        elif "cs precision" in l and "VALIDATION" in l:
            prec_res = l.split(":")[-1].replace(" ", "").split(",")
            prec_res = [float(c_r) for c_r in prec_res]
            cs_precs_euk[0].append(prec_res[0])
            cs_precs_euk[1].append(prec_res[1])
            cs_precs_euk[2].append(prec_res[2])
            cs_precs_euk[3].append(prec_res[3])

            cs_precs_neg[0].append(prec_res[4])
            cs_precs_neg[1].append(prec_res[5])
            cs_precs_neg[2].append(prec_res[6])
            cs_precs_neg[3].append(prec_res[7])

            cs_precs_pos[0].append(prec_res[8])
            cs_precs_pos[1].append(prec_res[9])
            cs_precs_pos[2].append(prec_res[10])
            cs_precs_pos[3].append(prec_res[11])

            cs_precs_arc[0].append(prec_res[12])
            cs_precs_arc[1].append(prec_res[13])
            cs_precs_arc[2].append(prec_res[14])
            cs_precs_arc[3].append(prec_res[15])
        elif "F1Score:" in l and "VALIDATION" in l:
            vals = l.split("F1Score:")[-1].replace(",","").split(" ")
            vals = [v for v in vals if v != '']
            vals = [float(v.replace("\n", "")) for v in vals]
            class_preds[0].append(float(vals[0]))
            class_preds[1].append(float(vals[1]))
            class_preds[2].append(float(vals[2]))
            class_preds[3].append(float(vals[3]))


    # fix for logs that have f1 score written as "precision"...
    if len(cs_precs_pos[0]) == 2 * len(cs_recalls_pos[0]):
        len_cs_rec = len(cs_recalls_pos[0])
        for j in range(4):
            cs_precs_euk[j], cs_precs_neg[j], cs_precs_pos[j], cs_precs_arc[j] = [cs_precs_euk[j][i * 2] for i in
                                                                                  range(len_cs_rec)], \
                                                                                 [cs_precs_neg[j][i * 2] for i in
                                                                                  range(len_cs_rec)], \
                                                                                 [cs_precs_pos[j][i * 2] for i in
                                                                                  range(len_cs_rec)], \
                                                                                 [cs_precs_arc[j][i * 2] for i in
                                                                                  range(len_cs_rec)]

    return euk_mcc, neg_mcc, pos_mcc, arc_mcc, train_loss, valid_loss, cs_recalls_euk, cs_recalls_neg, cs_recalls_pos, \
           cs_recalls_arc, cs_precs_euk, cs_precs_neg, cs_precs_pos, cs_precs_arc, class_preds


def remove_from_dictionary(res_dict, test_fld):
    tb_removed = pickle.load(open("../sp_data/sp6_partitioned_data_test_{}.bin".format(test_fld[0]), "rb"))
    tb_removed = tb_removed.keys()
    trimmed_res_dict = {}

    for seq, res in res_dict.items():
        if seq not in tb_removed:
            trimmed_res_dict[seq] = res
    return trimmed_res_dict


def extract_mean_test_results(run="param_search_0.2_2048_0.0001", result_folder="results_param_s_2/",
                              only_cs_position=False, remove_test_seqs=False, return_sptype_f1=False):
    full_dict_results = {}
    full_sptype_dict = {}
    epochs = []
    for tr_folds in [[0, 1], [1, 2], [0, 2]]:
        with open(result_folder + run + "_{}_{}.log".format(tr_folds[0], tr_folds[1]), "rt") as f:
            lines = f.readlines()
            try:
                epochs.append(int(lines[-2].split(" ")[2]))
            except:
                epochs.append(int(lines[-2].split(":")[-3].split(" ")[-1]))
    avg_epoch = np.mean(epochs)
    print("Results found on epochs: {}, {}, {}".format(*epochs))

    for tr_folds in [[0, 1], [1, 2], [0, 2]]:
        res_dict = pickle.load(open(result_folder + run + "_{}_{}_best.bin".format(tr_folds[0], tr_folds[1]), "rb"))
        if os.path.exists(result_folder + run + "_{}_{}_best_sptype.bin".format(tr_folds[0], tr_folds[1])):
            sptype_dict = pickle.load(open(result_folder + run + "_{}_{}_best_sptype.bin".format(tr_folds[0], tr_folds[1]), "rb"))
            full_sptype_dict.update(sptype_dict)
        else:
            full_sptype_dict = None
        test_fld = list({0, 1, 2} - set(tr_folds))
        if remove_test_seqs:
            full_dict_results.update(remove_from_dictionary(res_dict, test_fld))
        else:
            full_dict_results.update(res_dict)
    life_grp, seqs, true_lbls, pred_lbls = extract_seq_group_for_predicted_aa_lbls(filename="w_lg_w_glbl_lbl_100ep.bin",
                                                                                   dict_=full_dict_results)
    mccs, mccs2 = get_pred_accs_sp_vs_nosp(life_grp, seqs, true_lbls, pred_lbls, v=False, return_mcc2=True,
                                           sp_type="SP")
    # LIPO is SEC/SPII
    mccs_lipo, mccs2_lipo = get_pred_accs_sp_vs_nosp(life_grp, seqs, true_lbls, pred_lbls, v=False, return_mcc2=True,
                                                     sp_type="LIPO")
    # TAT is TAT/SPI
    mccs_tat, mccs2_tat = get_pred_accs_sp_vs_nosp(life_grp, seqs, true_lbls, pred_lbls, v=False, return_mcc2=True,
                                                   sp_type="TAT")
    if "param_search_w_nl_nh_0.0_4096_1e-05_4_4" in run:
        v = False
    else:
        v = False
    all_recalls, all_precisions, _, _, _, f1_scores = \
        get_cs_acc(life_grp, seqs, true_lbls, pred_lbls, v=v, only_cs_position=only_cs_position, sp_type="SP", sptype_preds=full_sptype_dict)
    all_recalls_lipo, all_precisions_lipo, _, _, _, f1_scores_lipo = \
        get_cs_acc(life_grp, seqs, true_lbls, pred_lbls, v=v, only_cs_position=only_cs_position, sp_type="LIPO", sptype_preds=full_sptype_dict )
    all_recalls_tat, all_precisions_tat, _, _, _, f1_scores_tat = \
        get_cs_acc(life_grp, seqs, true_lbls, pred_lbls, v=v, only_cs_position=only_cs_position, sp_type="TAT", sptype_preds=full_sptype_dict)
    if return_sptype_f1:
        return mccs, mccs2, mccs_lipo, mccs2_lipo, mccs_tat, mccs2_tat, all_recalls, all_precisions, all_recalls_lipo, \
               all_precisions_lipo, all_recalls_tat, all_precisions_tat, avg_epoch, f1_scores, f1_scores_lipo, f1_scores_tat, get_class_sp_accs(life_grp, seqs, true_lbls, pred_lbls)
    return mccs, mccs2, mccs_lipo, mccs2_lipo, mccs_tat, mccs2_tat, all_recalls, all_precisions, all_recalls_lipo, \
           all_precisions_lipo, all_recalls_tat, all_precisions_tat, avg_epoch, f1_scores, f1_scores_lipo, f1_scores_tat


def get_best_corresponding_eval_mcc(result_folder="results_param_s_2/", model="", metric="mcc"):
    tr_fold = [[0, 1], [1, 2], [0, 2]]
    all_best_mccs = []
    for t_f in tr_fold:
        with open(result_folder + model + "_{}_{}.log".format(t_f[0], t_f[1])) as f:
            lines = f.readlines()
        ep2mcc = {}
        best_mcc = -1
        for l in lines:
            if best_mcc != -1:
                continue
            elif "VALIDATION" in l and metric == "mcc":
                if "mcc" in l:
                    ep, mccs = l.split(":")[2], l.split(":")[4]
                    ep = int(ep.split("epoch")[-1].replace(" ", ""))
                    mccs = mccs.replace(" ", "").split(",")
                    mccs = np.mean([float(mcc) for mcc in mccs])
                    ep2mcc[ep] = mccs
            elif "train/validation loss" in l and metric == "loss":

                ep, mccs = l.split(":")[2], l.split(":")[3]
                ep = int(ep.split("epoch")[-1].split(" ")[1])
                if "," in mccs:
                    mccs = float(mccs.replace(" ", "").split(",")[0].split("/")[1])
                    ep2mcc[ep] = mccs / 2
                else:
                    mccs = float(mccs.replace(" ", "").split("/")[1])
                    ep2mcc[ep] = mccs
            elif "TEST" in l and "epoch" in l and "mcc" in l:
                best_ep = int(l.split(":")[2].split("epoch")[-1].replace(" ", ""))
                avg_last_5 = []
                for i in range(5):
                    best_mcc = ep2mcc[best_ep - i]
                    avg_last_5.append(best_mcc)
                best_mcc = np.mean(avg_last_5)
        all_best_mccs.append(best_mcc)
    return np.mean(all_best_mccs)


def get_mean_results_for_mulitple_runs(mdlind2mdlparams, mdl2results, plot_type="prec-rec", tol=1):
    avg_mcc, avg_mcc2, avg_mcc_lipo, avg_mcc2_lipo, avg_mccs_tat, avg_mccs2_tat, avg_prec, avg_recall, avg_prec_lipo, \
    avg_recall_lipo, avg_prec_tat, avg_recall_tat, no_of_mdls = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
    for ind, results in mdl2results.items():
        mccs, mccs2, mccs_lipo, mccs2_lipo, mccs_tat, mccs2_tat, all_recalls, all_precisions, \
        all_recalls_lipo, all_precisions_lipo, all_recalls_tat, all_precisions_tat, _ = results
        mdl = mdlind2mdlparams[ind].split("run_no")[0]
        if "patience_30" in mdl:
            mdl = "patience_30"
        else:
            mdl = "patience_60"
        if mdl in no_of_mdls:
            no_of_mdls[mdl] += 1
            avg_mcc[mdl].append(np.array(mccs))
            avg_recall[mdl].append(np.array(all_recalls))
            avg_prec[mdl].append(np.array(all_precisions))
            avg_mcc2[mdl].append(np.array(mccs2))
            avg_mcc_lipo[mdl].append(np.array(mccs_lipo))
            avg_mcc2_lipo[mdl].append(np.array(mccs2_lipo))
            avg_mccs_tat[mdl].append(np.array(avg_mccs_tat))
            avg_mccs2_tat[mdl].append(np.array(avg_mccs2_tat))
            avg_recall_lipo[mdl].append(np.array(all_recalls_lipo))
            avg_prec_lipo[mdl].append(np.array(all_precisions_lipo))
            avg_recall_tat[mdl].append(np.array(all_recalls_tat))
            avg_prec_tat[mdl].append(np.array(all_precisions_tat))
        else:
            no_of_mdls[mdl] = 1
            avg_mcc[mdl] = [np.array(mccs)]
            avg_recall[mdl] = [np.array(all_recalls)]
            avg_prec[mdl] = [np.array(all_precisions)]
            avg_mcc2[mdl] = [np.array(mccs2)]
            avg_mcc_lipo[mdl] = [np.array(mccs_lipo)]
            avg_mcc2_lipo[mdl] = [np.array(mccs2_lipo)]
            avg_mccs_tat[mdl] = [np.array(avg_mccs_tat)]
            avg_mccs2_tat[mdl] = [np.array(avg_mccs2_tat)]
            avg_recall_lipo[mdl] = [np.array(all_recalls_lipo)]
            avg_prec_lipo[mdl] = [np.array(all_precisions_lipo)]
            avg_prec_tat[mdl] = [np.array(all_precisions_tat)]
            avg_recall_tat[mdl] = [np.array(all_recalls_tat)]
    fig, axs = plt.subplots(2, 3, figsize=(12, 8)) if plot_type == "mcc" else plt.subplots(2, 4, figsize=(12, 8))
    plt.subplots_adjust(right=0.7)
    colors = ['red', 'green', 'orange', 'blue', 'brown', 'black']

    models = list(avg_mcc.keys())
    mdl2colors = {models[i]: colors[i] for i in range(len(models))}

    print(set(models))
    # TODO: Configure this for model names (usually full names are quite long and ugly)
    mdl2mdlnames = {}
    for mdl in models:
        if "patience_30" in mdl:
            mdl2mdlnames[mdl] = "patience 30"
        else:
            mdl2mdlnames[mdl] = "patience 60"
    # for mdl in models:
    #     if "lr_sched_searchlrsched_step_wrmpLrSched_0_" in mdl:
    #         mdl2mdlnames[mdl] = "step, 0 wrmp"
    #
    #     if "lr_sched_searchlrsched_expo_wrmpLrSched_10_" in mdl:
    #         mdl2mdlnames[mdl] = "expo, 10 wrmp"
    #
    #     if "lr_sched_searchlrsched_expo_wrmpLrSched_0_" in mdl:
    #         mdl2mdlnames[mdl] = "expo, 0 wrmp"
    #
    #     if "lr_sched_searchlrsched_step_wrmpLrSched_10_" in mdl:
    #         mdl2mdlnames[mdl] = "step, 10 wrmp"
    #
    #     if "test_beam_search" in mdl:
    #         mdl2mdlnames[mdl] = "no sched"
    # FOR GLBL LBL
    # for mdl in models:
    #     if "glbl_lbl_search_use_glbl_lbls_version_1_weight_1_" in mdl:
    #         mdl2mdlnames[mdl] = "version 1 weight 1"
    #     if "glbl_lbl_search_use_glbl_lbls_version_2_weight_1_" in mdl:
    #         mdl2mdlnames[mdl] = "version 2 weight 1"
    #     if "glbl_lbl_search_use_glbl_lbls_version_2_weight_0.1_" in mdl:
    #         mdl2mdlnames[mdl] = "version 2 weight 0.1"
    #     if "glbl_lbl_search_use_glbl_lbls_version_1_weight_0.1_" in mdl:
    #         mdl2mdlnames[mdl] = "version 1 weight 0.1"
    #     if "test_beam_search" in mdl:
    #         mdl2mdlnames[mdl] = "no glbl labels"
    plot_lgs = ['NEGATIVE', 'POSITIVE', 'ARCHAEA'] if plot_type == "mcc" else ['EUKARYA', 'NEGATIVE', 'POSITIVE',
                                                                               'ARCHAEA']
    for lg_ind, lg in enumerate(plot_lgs):
        if plot_type == "mcc":
            axs[0, 0].set_ylabel("MCC")
            axs[1, 0].set_ylabel("MCC2")
        else:
            axs[0, 0].set_ylabel("Recall")
            axs[1, 0].set_ylabel("Precision")
        for mdl in avg_mccs_tat.keys():
            x = np.linspace(0, 1, 1000)
            if plot_type == "mcc":
                kde = stats.gaussian_kde([avg_mcc[mdl][i][lg_ind + 1] for i in range(no_of_mdls[mdl])])
            else:
                kde = stats.gaussian_kde([avg_recall[mdl][i][lg_ind * 4 + tol] for i in range(no_of_mdls[mdl])])

            if lg_ind == 0:
                axs[0, lg_ind].plot(x, kde(x), color=mdl2colors[mdl])  # , label="{}".format(mdl2mdlnames[mdl]))
            else:
                axs[0, lg_ind].plot(x, kde(x), color=mdl2colors[
                    mdl])  # , label="Eukaryote {} param search".format(mdl2mdlnames[mdl]))
            axs[0, lg_ind].set_title(lg + " SEC/SPI")
        for mdl in avg_mccs_tat.keys():
            x = np.linspace(0, 1, 1000)
            if plot_type == "mcc":
                kde = stats.gaussian_kde([avg_mcc2[mdl][i][lg_ind + 1] for i in range(no_of_mdls[mdl])])
            else:
                kde = stats.gaussian_kde([avg_prec[mdl][i][lg_ind * 4 + tol] for i in range(no_of_mdls[mdl])])
            axs[1, lg_ind].plot(x, kde(x),
                                color=mdl2colors[mdl])  # , label="Eukaryote {} param search".format(mdl2mdlnames[mdl]))
            if lg_ind == len(plot_lgs) - 1:
                axs[1, lg_ind].plot(x, kde(x), color=mdl2colors[
                    mdl], label="{}".format(mdl2mdlnames[mdl]))

    plot_lgs = ['NEGATIVE', 'POSITIVE', 'ARCHAEA']
    plt.legend(loc=(1.04, 1))
    plt.show()

    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    plt.subplots_adjust(right=0.7)
    for lg_ind, lg in enumerate(['NEGATIVE', 'POSITIVE', 'ARCHAEA']):
        if plot_type == "mcc":
            axs[0, 0].set_ylabel("MCC")
            axs[1, 0].set_ylabel("MCC2")
        else:
            axs[0, 0].set_ylabel("Recall")
            axs[1, 0].set_ylabel("Precision")
        for mdl in avg_mccs_tat.keys():
            x = np.linspace(0, 1, 1000)
            if plot_type == "mcc":
                kde = stats.gaussian_kde([avg_mcc_lipo[mdl][i][lg_ind] for i in range(no_of_mdls[mdl])])
            else:
                kde = stats.gaussian_kde([avg_recall_lipo[mdl][i][lg_ind * 4 + tol] for i in range(no_of_mdls[mdl])])
            if lg_ind == 0:
                axs[0, lg_ind].plot(x, kde(x), color=mdl2colors[mdl])  # , label="{}".format(mdl2mdlnames[mdl]))
            else:
                axs[0, lg_ind].plot(x, kde(x), color=mdl2colors[
                    mdl])  # , label="Eukaryote {} param search".format(mdl2mdlnames[mdl]))
            axs[0, lg_ind].set_title(lg + " SEC/SPII")
        for mdl in avg_mccs_tat.keys():
            x = np.linspace(0, 1, 1000)
            if plot_type == "mcc":
                kde = stats.gaussian_kde([avg_mcc2_lipo[mdl][i][lg_ind] for i in range(no_of_mdls[mdl])])
            else:
                kde = stats.gaussian_kde([avg_prec_lipo[mdl][i][lg_ind * 4 + tol] for i in range(no_of_mdls[mdl])])
            axs[1, lg_ind].plot(x, kde(x),
                                color=mdl2colors[mdl])  # , label="Eukaryote {} param search".format(mdl2mdlnames[mdl]))
            if lg_ind == 2:
                axs[1, lg_ind].plot(x, kde(x), color=mdl2colors[
                    mdl], label="{}".format(mdl2mdlnames[mdl]))

    plt.legend(loc=(1.04, 1))
    plt.show()

    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    plt.subplots_adjust(right=0.7)
    for lg_ind, lg in enumerate(['NEGATIVE', 'POSITIVE', 'ARCHAEA']):
        if plot_type == "mcc":
            axs[0, 0].set_ylabel("MCC")
            axs[1, 0].set_ylabel("MCC2")
        else:
            axs[0, 0].set_ylabel("Recall")
            axs[1, 0].set_ylabel("Precision")
        for mdl in avg_mccs_tat.keys():
            x = np.linspace(0, 1, 1000)
            if plot_type == "mcc":
                kde = stats.gaussian_kde([avg_mccs_tat[mdl][i][lg_ind] for i in range(no_of_mdls[mdl])])
            else:
                kde = stats.gaussian_kde([avg_recall_tat[mdl][i][lg_ind * 4 + tol] for i in range(no_of_mdls[mdl])])
            if lg_ind == 0:
                axs[0, lg_ind].plot(x, kde(x), color=mdl2colors[mdl])  # , label="{}".format(mdl2mdlnames[mdl]))
            else:
                axs[0, lg_ind].plot(x, kde(x), color=mdl2colors[
                    mdl])  # , label="Eukaryote {} param search".format(mdl2mdlnames[mdl]))
            axs[0, lg_ind].set_title(lg + " TAT/SPI")
        for mdl in avg_mccs_tat.keys():
            x = np.linspace(0, 1, 1000)
            if plot_type == "mcc":
                kde = stats.gaussian_kde([avg_mccs2_tat[mdl][i][lg_ind] for i in range(no_of_mdls[mdl])])
            else:
                kde = stats.gaussian_kde([avg_prec_tat[mdl][i][lg_ind * 4 + tol] for i in range(no_of_mdls[mdl])])
            axs[1, lg_ind].plot(x, kde(x),
                                color=mdl2colors[mdl])  # , label="Eukaryote {} param search".format(mdl2mdlnames[mdl]))
            if lg_ind == 2:
                axs[1, lg_ind].plot(x, kde(x), color=mdl2colors[
                    mdl], label="{}".format(mdl2mdlnames[mdl]))

    plt.legend(loc=(1.04, 1))
    plt.show()
    if plot_type == "mcc":
        fig, axs = plt.subplots(1, 1, figsize=(4, 8))
        plt.subplots_adjust(right=0.7)
        for lg_ind, lg in enumerate(['EUKARYOTE']):
            axs[0, 0].set_ylabel("MCC")
            for mdl in avg_mccs_tat.keys():
                x = np.linspace(0, 1, 1000)
                kde = stats.gaussian_kde([avg_mcc[mdl][i][lg_ind + 1] for i in range(no_of_mdls[mdl])])
                axs[0, lg_ind].plot(x, kde(x), color=mdl2colors[mdl], label="{}".format(mdl2mdlnames[mdl]))
                axs[0, lg_ind].set_title(lg + " SEC/SPI")
        plt.legend(loc=(1.04, 1))
        plt.show()


def get_f1_scores(rec, prec):
    return [2 * rec[i] * prec[i] / (rec[i] + prec[i]) if rec[i] + prec[i] != 0 else 0 for i in range(len(rec))]


def extract_all_param_results(result_folder="results_param_s_2/", only_cs_position=False, compare_mdl_plots=False,
                              remove_test_seqs=False):
    sp6_recalls_sp1 = [0.747, 0.774, 0.808, 0.829, 0.639, 0.672, 0.689, 0.721, 0.800, 0.800, 0.800, 0.800, 0.500, 0.556,
                       0.556, 0.583]
    sp6_recalls_sp2 = [0.852, 0.852, 0.856, 0.864, 0.875, 0.883, 0.883, 0.883, 0.778, 0.778, 0.778, 0.778]
    sp6_recalls_tat = [0.706, 0.765, 0.784, 0.804, 0.556, 0.556, 0.667, 0.667, 0.333, 0.444, 0.444, 0.444]
    sp6_precs_sp1 = [0.661, 0.685, 0.715, 0.733, 0.534, 0.562, 0.575, 0.603, 0.632, 0.632, 0.632, 0.632, 0.643, 0.714,
                     0.714, 0.75]
    sp6_precs_sp2 = [0.913, 0.913, 0.917, 0.925, 0.929, 0.938, 0.938, 0.938, 0.583, 0.583, 0.583, 0.583]
    sp6_precs_tat = [0.679, 0.736, 0.755, 0.774, 0.714, 0.714, 0.857, 0.857, 0.375, 0.5, 0.5, 0.5]
    sp6_f1_sp1 = get_f1_scores(sp6_recalls_sp1, sp6_precs_sp1)
    sp6_f1_sp2 = get_f1_scores(sp6_precs_sp2, sp6_precs_sp2)
    sp6_f1_tat = get_f1_scores(sp6_recalls_tat, sp6_precs_tat)
    sp6_recalls_sp1 = [str(round(sp6_r_sp1, 2)) for sp6_r_sp1 in sp6_recalls_sp1]
    sp6_recalls_sp2 = [str(round(sp6_r_sp2, 2)) for sp6_r_sp2 in sp6_recalls_sp2]
    sp6_recalls_tat = [str(round(sp6_r_tat, 2)) for sp6_r_tat in sp6_recalls_tat]
    sp6_precs_sp1 = [str(round(sp6_prec_sp1, 2)) for sp6_prec_sp1 in sp6_precs_sp1]
    sp6_precs_sp2 = [str(round(sp6_p_sp2, 2)) for sp6_p_sp2 in sp6_precs_sp2]
    sp6_precs_tat = [str(round(sp6_p_tat, 2)) for sp6_p_tat in sp6_precs_tat]
    sp6_f1_sp1 = [str(round(sp6_f1_sp1_, 2)) for sp6_f1_sp1_ in sp6_f1_sp1]
    sp6_f1_sp2 = [str(round(sp6_f1_sp2_, 2)) for sp6_f1_sp2_ in sp6_f1_sp2]
    sp6_f1_tat = [str(round(sp6_f1_tat_, 2)) for sp6_f1_tat_ in sp6_f1_tat]
    files = os.listdir(result_folder)
    unique_params = set()
    for f in files:
        if "log" in f:
            # check if all 3 folds have finished
            dont_add = False
            for tr_f in [[0, 1], [1, 2], [0, 2]]:
                if "_".join(f.split("_")[:-2]) + "_{}_{}_best.bin".format(tr_f[0], tr_f[1]) not in files:
                    dont_add = True
            if not dont_add:
                unique_params.add("_".join(f.split("_")[:-2]))
    print(unique_params)
    mdl2results = {}
    mdlind2mdlparams = {}
    # order results by the eukaryote mcc
    eukaryote_mcc = []
    for ind, u_p in enumerate(unique_params):
        mccs, mccs2, mccs_lipo, mccs2_lipo, mccs_tat, mccs2_tat, \
        all_recalls, all_precisions, all_recalls_lipo, all_precisions_lipo, \
        all_recalls_tat, all_precisions_tat, avg_epoch, f1_scores, f1_scores_lipo, f1_scores_tat, f1_scores_sptype \
            = extract_mean_test_results(run=u_p, result_folder=result_folder,
                                        only_cs_position=only_cs_position,
                                        remove_test_seqs=remove_test_seqs, return_sptype_f1=True)
        all_recalls_lipo, all_precisions_lipo, \
        all_recalls_tat, all_precisions_tat, = list(np.reshape(np.array(all_recalls_lipo), -1)), list(
            np.reshape(np.array(all_precisions_lipo), -1)), \
                                               list(np.reshape(np.array(all_recalls_tat), -1)), list(
            np.reshape(np.array(all_precisions_tat), -1))
        mdl2results[ind] = (
        mccs, mccs2, mccs_lipo, mccs2_lipo, mccs_tat, mccs2_tat, list(np.reshape(np.array(all_recalls), -1)),
        list(np.reshape(np.array(all_precisions), -1)), all_recalls_lipo, all_precisions_lipo,
        all_recalls_tat, all_precisions_tat, f1_scores, f1_scores_lipo, f1_scores_tat, f1_scores_sptype, avg_epoch)
        mdlind2mdlparams[ind] = u_p
        eukaryote_mcc.append(get_best_corresponding_eval_mcc(result_folder, u_p))
    if compare_mdl_plots:
        get_mean_results_for_mulitple_runs(mdlind2mdlparams, mdl2results)
    best_to_worst_mdls = np.argsort(eukaryote_mcc)[::-1]
    for mdl_ind in best_to_worst_mdls:
        params = ""
        mdl_params = mdlind2mdlparams[mdl_ind]
        if "use_glbl_lbls" in mdl_params:
            params += "wGlbl"
        else:
            params += "nGlbl"
        # patience_ind = mdl_params.find("patience_") + len("patience_")
        # patience = mdl_params[patience_ind:patience_ind+2]
        # params += "_{}".format(patience)
        if "nlayers" in mdl_params:
            nlayers = mdl_params[mdl_params.find("nlayers") + len("nlayers"):].split("_")[1]
            params += "_{}".format(nlayers)
        if "nhead" in mdl_params:
            nhead = mdl_params[mdl_params.find("nhead") + len("nhead"):].split("_")[1]
            params += "_{}".format(nhead)
        if "lrsched" in mdl_params:
            lr_sched = mdl_params[mdl_params.find("lrsched") + len("lrsched"):].split("_")[1]
            params += "_{}".format(lr_sched)
        if "dos" in mdl_params:
            dos = mdl_params[mdl_params.find("dos"):].split("_")[1]
            params += "_{}".format(dos)
        mdlind2mdlparams[mdl_ind] = params

    print("\n\nMCC SEC/SPI TABLE\n\n")
    for mdl_ind in best_to_worst_mdls:
        mdl_params = " & ".join(mdlind2mdlparams[mdl_ind].split("_"))
        print(mdl_params, " & ", " & ".join([str(round(mcc, 3)) for mcc in mdl2results[mdl_ind][0]]), "&",
              " & ".join([str(round(mcc, 3)) for mcc in mdl2results[mdl_ind][1][1:]]), " & ",
              round(mdl2results[mdl_ind][-1], 3), "\\\\ \\hline")

    print("\n\nF1 table SEC/SPI\n\n")
    no_of_params = len(mdlind2mdlparams[best_to_worst_mdls[0]].split("_"))
    print(" SP6 ", " & " * no_of_params, " & ".join(sp6_f1_sp1), " & \\\\ \\hline")
    for mdl_ind in best_to_worst_mdls:
        print(" & ".join(mdlind2mdlparams[mdl_ind].split("_")), " & ",
              " & ".join([str(round(rec, 3)) for rec in np.concatenate(mdl2results[mdl_ind][-5])]), " & ",
              round(mdl2results[mdl_ind][-1], 3), "\\\\ \\hline")

    print("\n\nRecall table SEC/SPI\n\n")
    print(" SP6 ", " & " * no_of_params, " & ".join(sp6_recalls_sp1), " & \\\\ \\hline")
    for mdl_ind in best_to_worst_mdls:
        print(" & ".join(mdlind2mdlparams[mdl_ind].split("_")), " & ",
              " & ".join([str(round(rec, 3)) for rec in mdl2results[mdl_ind][6]]), " & ",
              round(mdl2results[mdl_ind][-1], 3), "\\\\ \\hline")

    print("\n\nPrec table SEC/SPI\n\n")
    print(" SP6 ", " & " * no_of_params, " & ".join(sp6_precs_sp1), " & \\\\ \\hline")
    for mdl_ind in best_to_worst_mdls:
        print(" & ".join(mdlind2mdlparams[mdl_ind].split("_")), " & ",
              " & ".join([str(round(rec, 3)) for rec in mdl2results[mdl_ind][7]]), "&",
              round(mdl2results[mdl_ind][-1], 3), "\\\\ \\hline")

    print("\n\nF1 table SEC/SPII \n\n")
    print(" SP6 ", " & " * no_of_params, " & ".join(sp6_f1_sp2), " & \\\\ \\hline")
    for mdl_ind in best_to_worst_mdls:
        print(" & ".join(mdlind2mdlparams[mdl_ind].split("_")), " & ",
              " & ".join([str(round(rec, 3)) for rec in np.concatenate(mdl2results[mdl_ind][-4])]), "&",
              round(mdl2results[mdl_ind][-1], 3), "\\\\ \\hline")

    print("\n\nRecall table SEC/SPII \n\n")
    print(" SP6 ", " & " * no_of_params, " & ".join(sp6_recalls_sp2), " & \\\\ \\hline")
    for mdl_ind in best_to_worst_mdls:
        print(" & ".join(mdlind2mdlparams[mdl_ind].split("_")), " & ",
              " & ".join([str(round(rec, 3)) for rec in mdl2results[mdl_ind][8]]), "&",
              round(mdl2results[mdl_ind][-1], 3), "\\\\ \\hline")

    print("\n\nPrec table SEC/SPII \n\n")
    print(" SP6 ", " & " * no_of_params, " & ".join(sp6_precs_sp2), " & \\\\ \\hline")
    for mdl_ind in best_to_worst_mdls:
        print(" & ".join(mdlind2mdlparams[mdl_ind].split("_")), " & ",
              " & ".join([str(round(rec, 3)) for rec in mdl2results[mdl_ind][9]]), "&",
              round(mdl2results[mdl_ind][-1], 3), "\\\\ \\hline")

    print("\n\nF1 table TAT/SPI \n\n")
    print(" SP6 ", " & " * no_of_params, " & ".join(sp6_f1_tat), " & \\\\ \\hline")
    for mdl_ind in best_to_worst_mdls:
        print(" & ".join(mdlind2mdlparams[mdl_ind].split("_")), " & ",
              " & ".join([str(round(rec, 3)) for rec in np.concatenate(mdl2results[mdl_ind][-3])]), "&",
              round(mdl2results[mdl_ind][-1], 3), "\\\\ \\hline")

    print("\n\nRecall table TAT/SPI \n\n")
    print(" SP6 ", " & " * no_of_params, " & ".join(sp6_recalls_tat), " & \\\\ \\hline")
    for mdl_ind in best_to_worst_mdls:
        print(" & ".join(mdlind2mdlparams[mdl_ind].split("_")), " & ",
              " & ".join([str(round(rec, 3)) for rec in mdl2results[mdl_ind][10]]), "&",
              round(mdl2results[mdl_ind][-1], 3), "\\\\ \\hline")

    print("\n\nPrec table TAT/SPI \n\n")
    print(" SP6 ", " & " * no_of_params, " & ".join(sp6_precs_tat), " & \\\\ \\hline")
    for mdl_ind in best_to_worst_mdls:
        print(" & ".join(mdlind2mdlparams[mdl_ind].split("_")), " & ",
              " & ".join([str(round(rec, 3)) for rec in mdl2results[mdl_ind][11]]), "&",
              round(mdl2results[mdl_ind][-1], 3), "\\\\ \\hline")

    print("\n\nMCC SEC/SPII TABLE\n\n")
    for mdl_ind in best_to_worst_mdls:
        print(" & ".join(mdlind2mdlparams[mdl_ind].split("_")), " & ",
              " & ".join([str(round(mcc, 3)) for mcc in mdl2results[mdl_ind][2]]), "&",
              " & ".join([str(round(mcc, 3)) for mcc in mdl2results[mdl_ind][3]]), "&",
              round(mdl2results[mdl_ind][-1], 3), "\\\\ \\hline")

    print("\n\nMCC TAT/SPI TABLE\n\n")
    for mdl_ind in best_to_worst_mdls:
        print(" & ".join(mdlind2mdlparams[mdl_ind].split("_")), " & ",
              " & ".join([str(round(mcc, 3)) for mcc in mdl2results[mdl_ind][4]]), "&",
              " & ".join([str(round(mcc, 3)) for mcc in mdl2results[mdl_ind][5]]), "&",
              round(mdl2results[mdl_ind][-1], 3), "\\\\ \\hline")

    print("\n\nSP-type preds F1\n\n")
    for mdl_ind in best_to_worst_mdls:
        print(" & ".join(mdlind2mdlparams[mdl_ind].split("_")), " & ",
              " & ".join([str(round(mcc, 3)) for mcc in mdl2results[mdl_ind][-2]]), "&",
              round(mdl2results[mdl_ind][-1], 3), "\\\\ \\hline")

    return mdl2results


def sanity_checks(run="param_search_0_2048_0.0001_", folder="results/"):
    # S = signal_peptide; T = Tat/SPI or Tat/SPII SP; L = Sec/SPII SP; P = SEC/SPIII SP; I = cytoplasm; M = transmembrane; O = extracellular;

    def get_last_contiguous_index(seq, signal_peptide):
        ind = 0
        while seq[ind] == signal_peptide and ind < len(seq) - 1:
            ind += 1
        return ind - 1

    def check_contiguous_sp(lbl_seqs):
        for l in lbl_seqs:
            signal_peptide = None
            l = l.replace("ES", "")
            if "S" in l:
                signal_peptide = "S"
            elif "T" in l:
                signal_peptide = "T"
            elif "L" in l:
                signal_peptide = "L"
            elif "P" in l:
                signal_peptide = "P"
            if signal_peptide is not None:

                if l.rfind(signal_peptide) != get_last_contiguous_index(l, signal_peptide):
                    print(l, l.rfind(signal_peptide), get_last_contiguous_index(l, signal_peptide), signal_peptide)

    for tr_fold in [[0, 1], [1, 2], [0, 2]]:
        labels = pickle.load(open(folder + run + "{}_{}.bin".format(tr_fold[0], tr_fold[1]), "rb")).values()
        check_contiguous_sp(labels)


def extract_all_mdl_results(mdl2results):
    euk_mcc, neg_mcc, pos_mcc, arch_mcc = [], [], [], []
    euk_rec, neg_rec, pos_rec, arch_rec = [[], [], [], []], [[], [], [], []], [[], [], [], []], [[], [], [], []]
    euk_prec, neg_prec, pos_prec, arch_prec = [[], [], [], []], [[], [], [], []], [[], [], [], []], [[], [], [], []]
    for _, (mccs, recalls, precisions, epochs) in mdl2results.items():
        euk_mcc.append(mccs[0])
        neg_mcc.append(mccs[1])
        pos_mcc.append(mccs[2])
        arch_mcc.append(mccs[3])
        euk_rec[0].append(recalls[0])
        euk_rec[1].append(recalls[1])
        euk_rec[2].append(recalls[2])
        euk_rec[3].append(recalls[3])
        neg_rec[0].append(recalls[4])
        neg_rec[1].append(recalls[5])
        neg_rec[2].append(recalls[6])
        neg_rec[3].append(recalls[7])
        pos_rec[0].append(recalls[8])
        pos_rec[1].append(recalls[9])
        pos_rec[2].append(recalls[10])
        pos_rec[3].append(recalls[11])
        arch_rec[0].append(recalls[12])
        arch_rec[1].append(recalls[13])
        arch_rec[2].append(recalls[14])
        arch_rec[3].append(recalls[15])

        euk_prec[0].append(precisions[0])
        euk_prec[1].append(precisions[1])
        euk_prec[2].append(precisions[2])
        euk_prec[3].append(precisions[3])
        neg_prec[0].append(precisions[4])
        neg_prec[1].append(precisions[5])
        neg_prec[2].append(precisions[6])
        neg_prec[3].append(precisions[7])
        pos_prec[0].append(precisions[8])
        pos_prec[1].append(precisions[9])
        pos_prec[2].append(precisions[10])
        pos_prec[3].append(precisions[11])
        arch_prec[0].append(precisions[12])
        arch_prec[1].append(precisions[13])
        arch_prec[2].append(precisions[14])
        arch_prec[3].append(precisions[15])
    return euk_mcc, neg_mcc, pos_mcc, arch_mcc, euk_rec, neg_rec, pos_rec, arch_rec, euk_prec, neg_prec, pos_prec, arch_prec


def visualize_training_variance(mdl2results, mdl2results_hps=None):
    def plot_4_figs(measures, sp6_measure, hp_s_measures=None, name="", plot_hps=False):
        euk, neg, pos, arch = measures
        if plot_hps:
            euk_hps, neg_hps, pos_hps, arch_hps = hp_s_measures

        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(name)
        x = np.linspace(0, 1, 1000)
        kde = stats.gaussian_kde(euk)
        # axs[0, 0].plot(x, kde(x), color = "blue", label="Eukaryote {}".format(name))
        axs[0, 0].hist(euk, color="blue", bins=10, label="Eukaryote {}".format(name))
        if plot_hps:
            kde = stats.gaussian_kde(euk_hps)
            axs[0, 0].hist(euk_hps, alpha=0.5, color="red", bins=10, label="Eukaryote {} param search".format(name))
            # axs[0, 0].plot(x, kde(x), color = "red", label="Eukaryote {} param search".format(name))
        axs[0, 0].set_ylabel("No of models")
        axs[0, 0].set_xlim(0, 1)
        axs[0, 0].plot([sp6_measure[0], sp6_measure[0]], [0, 5], 'r--', label="SP6 result")
        axs[0, 0].legend()

        kde = stats.gaussian_kde(neg)
        axs[0, 1].hist(neg, bins=10, color="blue", label="Negative {}".format(name))
        # axs[0, 1].plot(x, kde(x), color = "blue", label="Negative {}".format(name))
        if plot_hps:
            kde = stats.gaussian_kde(neg_hps)
            axs[0, 1].hist(neg_hps, alpha=0.5, bins=10, color="red", label="Negative {} param search".format(name))
            # axs[0, 1].plot(x, kde(x), color = "red", label="Negative {} param search".format(name))
        axs[0, 1].set_xlim(0, 1)
        axs[0, 1].plot([sp6_measure[1], sp6_measure[1]], [0, 5], 'r--', label="SP6 result")

        axs[0, 1].legend()

        # axs[1, 0].hist(pos,bins=10,color = "blue", label="Positive {}".format(name))
        kde = stats.gaussian_kde(pos)
        axs[1, 0].hist(pos, bins=10, color="blue", label="Positive {}".format(name))
        # axs[1, 0].plot(x, kde(x), color = "blue", label="Positive {}".format(name))

        if plot_hps:
            # kde = stats.gaussian_kde(pos_hps)
            axs[1, 0].hist(pos_hps, color="red", alpha=0.5, bins=10, label="Positive {} param search".format(name))
            axs[1, 0].plot(x, kde(x), color="red", label="Positive {} param search".format(name))

        axs[1, 0].set_xlim(0, 1)
        axs[1, 0].set_ylabel("No of models")
        axs[1, 0].plot([sp6_measure[2], sp6_measure[2]], [0, 5], 'r--', label="SP6 result")
        axs[1, 0].legend()
        axs[1, 0].set_xlabel(name.split(" ")[0])

        kde = stats.gaussian_kde(arch)
        axs[1, 1].hist(arch, bins=10, color="blue", label="Archaea {}".format(name))
        # axs[1, 1].plot(x, kde(x), color = "blue",label="Archaea {}".format(name))
        if plot_hps:
            kde = stats.gaussian_kde(arch_hps)
            axs[1, 1].hist(arch_hps, alpha=0.5, color="red", bins=10, label="Archaea {} param search".format(name))
            # axs[1, 1].plot(x, kde(x), color = "red", label="Archaea {} param search".format(name))
        axs[1, 1].set_xlim(0, 1)
        axs[1, 1].plot([sp6_measure[3], sp6_measure[3]], [0, 5], 'r--', label="SP6 result")

        axs[1, 1].legend()
        axs[1, 1].set_xlabel(name.split(" ")[0])
        plt.show()
        # plt.savefig("/home/alex/Desktop/sp6_ds_transformer_nmt_results/{}_{}.png".format(name, "mcc"))

    euk_mcc_sp6, neg_mcc_sp6, pos_mcc_sp6, arch_mcc_sp6 = 0.868, 0.811, 0.878, 0.737
    euk_rec_sp6, neg_rec_sp6, pos_rec_sp6, arch_rec_sp6 = [0.747, 0.774, 0.808, 0.829], [0.639, 0.672, 0.689, 0.721], \
                                                          [0.800, 0.800, 0.800, 0.800], [0.500, 0.556, 0.556, 0.583]
    euk_prec_sp6, neg_prec_sp6, pos_prec_sp6, arch_prec_sp6 = [0.661, 0.685, 0.715, 0.733], [0.534, 0.562, 0.575,
                                                                                             0.603], \
                                                              [0.632, 0.632, 0.632, 0.632], [0.643, 0.714, 0.714, 0.75]
    euk_mcc, neg_mcc, pos_mcc, arch_mcc, euk_rec, neg_rec, \
    pos_rec, arch_rec, euk_prec, neg_prec, pos_prec, arch_prec = extract_all_mdl_results(mdl2results)
    if mdl2results_hps is not None:
        euk_hps_mcc, neg_hps_mcc, pos_hps_mcc, arch_hps_mcc, euk_hps_rec, neg_hps_rec, pos_hps_rec, arch_hps_rec, \
        euk_hps_prec, neg_hps_prec, pos_hps_prec, arch_hps_prec = extract_all_mdl_results(mdl2results_hps)
    else:
        euk_hps_mcc, neg_hps_mcc, pos_hps_mcc, arch_hps_mcc, euk_hps_rec, neg_hps_rec, pos_hps_rec, arch_hps_rec, \
        euk_hps_prec, neg_hps_prec, pos_hps_prec, arch_hps_prec = None, None, None, None, [None, None, None, None], \
                                                                  [None, None, None, None], [None, None, None, None], \
                                                                  [None, None, None, None], [None, None, None, None], \
                                                                  [None, None, None, None], [None, None, None, None], \
                                                                  [None, None, None, None]
    plot_hps = mdl2results_hps is not None
    plot_4_figs([euk_mcc, neg_mcc, pos_mcc, arch_mcc],
                [euk_mcc_sp6, neg_mcc_sp6, pos_mcc_sp6, arch_mcc_sp6],
                [euk_hps_mcc, neg_hps_mcc, pos_hps_mcc, arch_hps_mcc],
                name='mcc', plot_hps=plot_hps)
    for i in range(4):
        plot_4_figs([euk_rec[i], neg_rec[i], pos_rec[i], arch_rec[i]],
                    [euk_rec_sp6[i], neg_rec_sp6[i], pos_rec_sp6[i], arch_rec_sp6[i]],
                    [euk_hps_rec[i], neg_hps_rec[i], pos_hps_rec[i], arch_hps_rec[i]],
                    name='recall tol={}'.format(i), plot_hps=plot_hps)
        plot_4_figs([euk_prec[i], neg_prec[i], pos_prec[i], arch_prec[i]],
                    [euk_prec_sp6[i], neg_prec_sp6[i], pos_prec_sp6[i], arch_prec_sp6[i]],
                    [euk_hps_prec[i], neg_hps_prec[i], pos_hps_prec[i], arch_hps_prec[i]],
                    name='precision tol={}'.format(i), plot_hps=plot_hps)


def extract_calibration_probs_for_mdl(model="parameter_search_patience_60use_glbl_lbls_use_glbl_lbls_versio"
                                            "n_1_weight_0.1_lr_1e-05_nlayers_3_nhead_16_lrsched_none_trFlds_",
                                      folder='huge_param_search/patience_60/'):
    all_lg, all_seqs, all_tl, all_pred_lbls, sp2probs = [], [], [], [], {}
    for tr_f in [[0, 1], [0, 2], [1, 2]]:
        prob_file = "{}{}_{}_best_sp_probs.bin".format(model, tr_f[0], tr_f[1])
        preds_file = "{}{}_{}_best.bin".format(model, tr_f[0], tr_f[1])
        life_grp, seqs, true_lbls, pred_lbls = extract_seq_group_for_predicted_aa_lbls(filename=folder + preds_file)
        all_lg.extend(life_grp)
        all_seqs.extend(seqs)
        all_tl.extend(true_lbls)
        all_pred_lbls.extend(pred_lbls)
        sp2probs.update(pickle.load(open(folder + prob_file, "rb")))
    get_prob_calibration_and_plot("", all_lg, all_seqs, all_tl, all_pred_lbls, sp2probs=sp2probs)


def duplicate_Some_logs():
    from subprocess import call

    files = os.listdir("beam_test")
    for f in files:
        if "log" in f:
            file = f.replace(".bin", "_best.bin")
            cmd = ["cp", "beam_test/" + f, "beam_test/actual_beams/best_beam_" + file]
            call(cmd)
    exit(1)


if __name__ == "__main__":
    # extract_calibration_probs_for_mdl()
    # duplicate_Some_logs()
    # exit(1)
    mdl2results = extract_all_param_results(only_cs_position=False,
                                            result_folder="separate-glbl_large_separate_save_long/",
                                            compare_mdl_plots=False,
                                            remove_test_seqs=False)
    mdl2results = extract_all_param_results(only_cs_position=False,
                                            result_folder="separate-glbl_rerun_separate_save_long_run/",
                                            compare_mdl_plots=False,
                                            remove_test_seqs=False)
    mdl2results = extract_all_param_results(only_cs_position=False,
                                            result_folder="separate-glbl_weighted_loss_separat/",
                                            compare_mdl_plots=False,
                                            remove_test_seqs=False)
    visualize_validation(run="rerun_separate_save_long_run_", folds=[0, 1], folder="separate-glbl_rerun_separate_save_long_run/")
    visualize_validation(run="weighted_loss_separate_save_long_run_", folds=[0, 1], folder="separate-glbl_weighted_loss_separat/")

    mdl2results = extract_all_param_results(only_cs_position=False,
                                            result_folder="separate-glbl_rerun_separate_save_long_run/only_cs/",
                                            compare_mdl_plots=False,
                                            remove_test_seqs=False)
    mdl2results = extract_all_param_results(only_cs_position=False,
                                            result_folder="separate-glbl_save_long_run/",
                                            compare_mdl_plots=False,
                                            remove_test_seqs=False)
    visualize_validation(run="rerun_3_16_validate_on_mcc2_drop_separate_glbl_cs_", folds=[1, 2], folder="separate-glbl_rerun_best/")
    mdl2results = extract_all_param_results(only_cs_position=False,
                                            result_folder="separate-glbl_large_separate_save_long/",
                                            compare_mdl_plots=False,
                                            remove_test_seqs=False)
    visualize_validation(run="large_separate_save_long_run_", folds=[1, 2], folder="separate-glbl_large_separate_save_long/")
    visualize_validation(run="scale_input_linear_pos_enc_separate_saves_", folds=[1, 2], folder="separate-glbl_scale_input_linear/")
    visualize_validation(run="cnn3_3_32_validate_on_mcc2_drop_separate_glbl_cs_", folds=[0, 1], folder="separate-glbl_3_32_mdl/")
    visualize_validation(run="linear_pos_enc_separate_saves_", folds=[0, 1], folder="separate-glbl_linear_pos_enc/")
    mdl2results = extract_all_param_results(only_cs_position=False,
                                            result_folder="separate-glbl_linear_pos_enc/",
                                            compare_mdl_plots=False,
                                            remove_test_seqs=False)
    mdl2results = extract_all_param_results(only_cs_position=False,
                                            result_folder="separate-glbl_no_pos_enc/",
                                            compare_mdl_plots=False,
                                            remove_test_seqs=False)
    visualize_validation(run="cnn3_3_16_validate_on_mcc2_drop_separate_glbl_cs_", folds=[1, 2], folder="separate-glbl_cnn3/")
    mdl2results = extract_all_param_results(only_cs_position=False,
                                            result_folder="separate-glbl_patience_swa/",
                                            compare_mdl_plots=False,
                                            remove_test_seqs=False)
    visualize_validation(run="large3_validate_on_mcc2_drop_separate_glbl_cs_", folds=[0, 1], folder="separate-glbl_large3/")
    visualize_validation(run="patience_swa_model_", folds=[0, 1], folder="separate-glbl_patience_swa/")
    visualize_validation(run="large3_validate_on_mcc2_drop_separate_glbl_cs_", folds=[0, 1], folder="separate-glbl_large3/")


    mdl2results = extract_all_param_results(only_cs_position=False,
                                            result_folder="separate-glbl_large2_01drop_mdl/",
                                            compare_mdl_plots=False,
                                            remove_test_seqs=False)
    visualize_validation(run="input_drop_validate_on_mcc2_drop_separate_glbl_cs_", folds=[1, 2],
                         folder="separate-glbl_input_drop/")
    mdl2results = extract_all_param_results(only_cs_position=False,
                                            result_folder="separate-glbl_input_drop/",
                                            compare_mdl_plots=False,
                                            remove_test_seqs=False)

    visualize_validation(run="tune_cs_fromstart_v2_folds_", folds=[0, 1], folder="tune_cs_from_start/")
    mdl2results = extract_all_param_results(only_cs_position=False,
                                            result_folder="tune_cs_from_start/",
                                            compare_mdl_plots=False,
                                            remove_test_seqs=False)

    visualize_validation(run="validate_on_mcc2_drop_separate_glbl_cs_", folds=[1, 2], folder="separate-glbl-mcc2-drop/")
    visualize_validation(run="tune_cs_run_", folds=[0, 1], folder="tune_cs_test/")

    mdl2results = extract_all_param_results(only_cs_position=False,
                                            result_folder="separate-glbl-mcc2/",
                                            compare_mdl_plots=False,
                                            remove_test_seqs=False)

    # visualize_validation(run="separate_glbl_cs_", folds=[0, 1],folder="separate-glbl/")
    visualize_validation(run="validate_on_mcc_separate_glbl_cs_", folds=[0, 1], folder="separate-glbl-mcc/")

    # mdl2results = extract_all_param_results(only_cs_position=False, result_folder="drop_large_crct_v2_max_glbl_lg_deailed_sp_v1/",
    #                                         compare_mdl_plots=False,
    #                                         remove_test_seqs=False)

    visualize_validation(run="crct_v2_max_glbl_lg_deailed_sp_v1_", folds=[0, 1], folder="crct_simplified_glblv2_max/")
    visualize_validation(run="parameter_search_patience_30lr_1e-05_nlayers_3_nhead_16_lrsched_step_trFlds_",
                         folds=[0, 1], folder="huge_param_search/")
    visualize_validation(run="crct_v2_max_glbl_lg_deailed_sp_v1_", folds=[0, 1], folder="crct_simplified_glblv2_max/")
    visualize_validation(run="glbl_lg_deailed_sp_v1_", folds=[0, 1], folder="glbl_deailed_sp_v1/")

    visualize_validation(run="wdrop_noglbl_val_on_test_", folds=[1, 2], folder="wlg10morepatience/")
    visualize_validation(run="wdrop_noglbl_val_on_test_", folds=[0, 2], folder="wlg10morepatience/")
    # print("huh?")
    # mdl2results = extract_all_param_results(only_cs_position=False, result_folder="results_param_s_2/")
    # mdl2results_hps = extract_all_param_results(only_cs_position=False, result_folder="results_param_s_2/")
    # visualize_training_variance(mdl2results)#, mdl2results_hps)
    # extract_mean_test_results(run="param_search_0_2048_1e-05")
    # sanity_checks()
    # visualize_validation(run="param_search_0.2_4096_1e-05_", folds=[0,2])
    # visualize_validation(run="param_search_0.2_4096_1e-05_", folds=[1,2])
    # life_grp, seqs, true_lbls, pred_lbls = extract_seq_group_for_predicted_aa_lbls(filename="w_lg_w_glbl_lbl_100ep.bin")
    # sp_pred_accs = get_pred_accs_sp_vs_nosp(life_grp, seqs, true_lbls, pred_lbls,v=True)
    # all_recalls, all_precisions, total_positives, false_positives, predictions = get_cs_acc(life_grp, seqs, true_lbls, pred_lbls)
