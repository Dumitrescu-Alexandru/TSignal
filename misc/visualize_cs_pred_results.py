from scipy import stats
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef as compute_mcc
import os
import pickle
from Bio import SeqIO
import numpy as np


def get_cs_acc(life_grp, seqs, true_lbls, pred_lbls, v=False, only_cs_position=False):
    def is_cs(predicted_cs_ind, true_lbl_seq):
        if true_lbl_seq[predicted_cs_ind] == "S" and true_lbl_seq[predicted_cs_ind + 1] != "S":
            return True
        return False

    def get_acc_for_tolerence(ind, t_lbl):
        true_cs = 0
        while t_lbl[true_cs] == "S" and true_cs < len(t_lbl):
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
            return np.array([0, 0, 0, 0, 1, 0, 1, 1, 1, 1])
        else:
            # if ind==0, SP was not even predicted (so there is no CS prediction) and this affects precision metric
            # tp/(tp+fp). It means this isnt a tp or a fp, its a fn
            return np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0])

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
    for l, s, t, p in zip(life_grp, seqs, true_lbls, pred_lbls):
        life_grp, sp_info = l.split("|")
        ind = 0
        predicted_sp = p[0]
        is_sp = predicted_sp in sp_types
        if sp_info == "SP":

            while (p[ind] == "S" or (p[ind] == predicted_sp and is_sp and only_cs_position)) and ind < len(p) - 1:
                # when only_cs_position=True, the cleavage site positions will be taken into account irrespective of
                # whether the predicted SP is the correct kind
                ind += 1
            predictions[grp2_ind[life_grp]] += get_acc_for_tolerence(ind, t)

        elif sp_info != "SP" and p[ind] == "S":
            predictions[grp2_ind[life_grp]] += np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
    if v:
        print(" count_tol_fn, count_complete_fn, count_otherSPpred", count_tol_fn, count_complete_fn, count_otherSPpred)
    all_recalls = []
    all_precisions = []
    total_positives = []
    false_positives = []
    for life_grp, ind in grp2_ind.items():
        current_preds = predictions[grp2_ind[life_grp]]
        if v:
            print("Recall {}: {}".format(life_grp, [current_preds[i] / current_preds[4] for i in range(4)]))
            print("Prec {}: {}".format(life_grp,
                                       [current_preds[i] / (current_preds[i] + current_preds[5]) for i in range(4)]))
        all_recalls.append([current_preds[i] / current_preds[4] for i in range(4)])
        all_precisions.append([])
        for i in range(4):
            if current_preds[5] + current_preds[i] == 0:
                all_precisions[-1].append(0.)
            else:
                all_precisions[-1].append(
                    current_preds[i] / (current_preds[-1] + current_preds[i] + current_preds[i + 6]))
        total_positives.append(current_preds[4])
        false_positives.append(current_preds[5])
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
                                    , predictions[grp2_ind[grp]][1]))
        if v:
            print("{}: {}".format(grp, mccs[-1]))

    return mccs


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


def get_cs_and_sp_pred_results(filename="run_wo_lg_info.bin", v=False):
    life_grp, seqs, true_lbls, pred_lbls = extract_seq_group_for_predicted_aa_lbls(filename=filename)
    sp_pred_accs = get_pred_accs_sp_vs_nosp(life_grp, seqs, true_lbls, pred_lbls, v=v)
    all_recalls, all_precisions, total_positives, false_positives, predictions = get_cs_acc(life_grp, seqs, true_lbls,
                                                                                            pred_lbls, v=v)
    return sp_pred_accs, all_recalls, all_precisions, total_positives, false_positives, predictions


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
    plt.savefig("/home/alex/Desktop/sp6_ds_transformer_nmt_results/" + name + "loss.png")


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
    plt.savefig("/home/alex/Desktop/sp6_ds_transformer_nmt_results/{}_{}.png".format(name, "mcc"))


def extract_and_plot_prec_recall(results, metric="recall", name="param_search_0.2_2048_0.0001_"):
    cs_res_euk, cs_res_neg, cs_res_pos, cs_res_arc = results
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    for i in range(4):
        axs[0, 0].plot(cs_res_euk[i], label="Eukaryote {} tol={}".format(metric, i))
        axs[0, 0].set_ylabel(metric)
        axs[0, 0].legend()
        axs[0, 0].set_ylim(-0.1, 1.1)

        axs[0, 1].plot(cs_res_neg[i], label="Negative {} tol={}".format(metric, i))
        axs[0, 1].legend()
        axs[0, 1].set_ylim(-0.1, 1.1)

        axs[1, 0].plot(cs_res_pos[i], label="Positive {} tol={}".format(metric, i))
        axs[1, 0].legend()
        axs[1, 0].set_xlabel("epochs")
        axs[1, 0].set_ylim(-0.1, 1.1)
        axs[1, 0].set_ylabel(metric)

        axs[1, 1].plot(cs_res_arc[i], label="Archaea {} tol={}".format(metric, i))
        axs[1, 1].legend()
        axs[1, 1].set_xlabel("epochs")

        axs[1, 1].set_ylim(-0.1, 1.1)

    plt.savefig("/home/alex/Desktop/sp6_ds_transformer_nmt_results/{}_{}.png".format(name, metric))


def visualize_validation(run="param_search_0.2_2048_0.0001_", folds=[0, 1], folder=""):
    all_results = []
    euk_mcc, neg_mcc, pos_mcc, arc_mcc, train_loss, valid_loss, cs_recalls_euk, cs_recalls_neg, cs_recalls_pos, \
    cs_recalls_arc, cs_precs_euk, cs_precs_neg, cs_precs_pos, cs_precs_arc = extract_results(run, folds=folds,
                                                                                             folder=folder)
    plot_mcc([euk_mcc, neg_mcc, pos_mcc, arc_mcc], name=run)
    plot_losses([train_loss, valid_loss], name=run)
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

    return euk_mcc, neg_mcc, pos_mcc, arc_mcc, train_loss, valid_loss, cs_recalls_euk, cs_recalls_neg, cs_recalls_pos, \
           cs_recalls_arc, cs_precs_euk, cs_precs_neg, cs_precs_pos, cs_precs_arc


def extract_mean_test_results(run="param_search_0.2_2048_0.0001", result_folder="results_param_s_2/",
                              only_cs_position=False):
    full_dict_results = {}
    epochs = []
    for tr_folds in [[0, 1], [1, 2], [0, 2]]:
        with open(result_folder + run + "_{}_{}.log".format(tr_folds[0], tr_folds[1]), "rt") as f:
            lines = f.readlines()
            epochs.append(int(lines[-2].split(" ")[2]))
    avg_epoch = np.mean(epochs)
    print("Results found on epochs: {}, {}, {}".format(*epochs))

    for tr_folds in [[0, 1], [1, 2], [0, 2]]:
        res_dict = pickle.load(open(result_folder + run + "_{}_{}_best.bin".format(tr_folds[0], tr_folds[1]), "rb"))
        full_dict_results.update(res_dict)
    life_grp, seqs, true_lbls, pred_lbls = extract_seq_group_for_predicted_aa_lbls(filename="w_lg_w_glbl_lbl_100ep.bin",
                                                                                   dict_=full_dict_results)
    mccs = get_pred_accs_sp_vs_nosp(life_grp, seqs, true_lbls, pred_lbls, v=False)
    if "param_search_w_nl_nh_0.0_4096_1e-05_4_4" in run:
        v = False
    else:
        v = False
    all_recalls, all_precisions, _, _, _ = \
        get_cs_acc(life_grp, seqs, true_lbls, pred_lbls, v=v, only_cs_position=only_cs_position)
    return mccs, all_recalls, all_precisions, avg_epoch


def get_best_corresponding_eval_mcc(result_folder="results_param_s_2/", model=""):
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
            elif "VALIDATION" in l and "mcc" in l:
                ep, mccs = l.split(":")[2], l.split(":")[4]
                ep = int(ep.split("epoch")[-1].replace(" ", ""))
                mccs = mccs.replace(" ", "").split(",")
                mccs = [float(mcc) for mcc in mccs]
                ep2mcc[ep] = mccs
            elif "TEST" in l and "epoch" in l and "mcc" in l:
                best_ep = int(l.split(":")[2].split("epoch")[-1].replace(" ", ""))
                best_mcc = ep2mcc[best_ep]
        all_best_mccs.append(best_mcc)
    return np.mean(all_best_mccs)


def get_mean_results_for_mulitple_runs(mdlind2mdlparams, mdl2results):
    avg_mcc, avg_prec, avg_recall, no_of_mdls = {}, {}, {}, {}
    for ind, results in mdl2results.items():
        mccs, all_recalls, all_precisions, _ = results
        mdl = mdlind2mdlparams[ind].split("run_no")[0]
        if mdl in no_of_mdls:
            no_of_mdls[mdl] += 1
            avg_mcc[mdl] += np.array(mccs)
            avg_recall[mdl] += np.array(all_recalls)
            avg_prec[mdl] += np.array(all_precisions)
        else:
            no_of_mdls[mdl] = 1
            avg_mcc[mdl] = np.array(mccs)
            avg_recall[mdl] = np.array(all_recalls)
            avg_prec[mdl] = np.array(all_precisions)
    for mdl, no_of_tests in no_of_mdls.items():
        print(mdl)
        print(avg_mcc[mdl]/no_of_tests)
        print(avg_recall[mdl]/no_of_tests)
        print(avg_prec[mdl]/no_of_tests)


def extract_all_param_results(result_folder="results_param_s_2/", only_cs_position=False):
    files = os.listdir(result_folder)
    unique_params = set()
    for f in files:
        if "log" in f:
            unique_params.add("_".join(f.split("_")[:-2]))
    mdl2results = {}
    mdlind2mdlparams = {}
    # order results by the eukaryote mcc
    eukaryote_mcc = []
    for ind, u_p in enumerate(unique_params):
        print(u_p)
        mccs, all_recalls, all_precisions, avg_epoch = extract_mean_test_results(run=u_p, result_folder=result_folder,
                                                                                 only_cs_position=only_cs_position)
        mdl2results[ind] = (mccs, list(np.reshape(np.array(all_recalls), -1)),
                            list(np.reshape(np.array(all_precisions), -1)), avg_epoch)
        mdlind2mdlparams[ind] = u_p
        eukaryote_mcc.append(get_best_corresponding_eval_mcc(result_folder, u_p))
    get_mean_results_for_mulitple_runs(mdlind2mdlparams, mdl2results)
    best_to_worst_mdls = np.argsort(eukaryote_mcc)[::-1]
    print("\n\nMCC TABLE\n\n")
    for mdl_ind in best_to_worst_mdls:
        print(" & ".join(mdlind2mdlparams[mdl_ind].split("_")[6:]), "&", " & ".join([str(round(mcc, 3))
                                                                                     for mcc in
                                                                                     mdl2results[mdl_ind][0]]), "&",
              round(mdl2results[mdl_ind][-1], 3), "\\\\ \\hline")

    print("\n\nRecall table \n\n")
    for mdl_ind in best_to_worst_mdls:
        print(" & ".join(mdlind2mdlparams[mdl_ind].split("_")[6:]), "&",
              " & ".join([str(round(rec, 3)) for rec in mdl2results[mdl_ind][1]]), "&",
              round(mdl2results[mdl_ind][-1], 3), "\\\\ \\hline")

    print("\n\nPrec table \n\n")
    for mdl_ind in best_to_worst_mdls:
        print(" & ".join(mdlind2mdlparams[mdl_ind].split("_")[6:]), "&",
              " & ".join([str(round(rec, 3)) for rec in mdl2results[mdl_ind][2]]), "&",
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


if __name__ == "__main__":
    mdl2results = extract_all_param_results(only_cs_position=False, result_folder="lr_sched_search/")
    # mdl2results = extract_all_param_results(only_cs_position=False, result_folder="results_param_s_2/")
    # mdl2results_hps = extract_all_param_results(only_cs_position=False, result_folder="results_param_s_2/")
    # visualize_training_variance(mdl2results)#, mdl2results_hps)
    # extract_mean_test_results(run="param_search_0_2048_1e-05")
    # sanity_checks()
    # visualize_validation(run="param_search_patience_60_w_nl_nh_0.0_4096_1e-05_2_8__folds_", folds=[0,1],folder="results_param_search_patience_60/")
    # visualize_validation(run="param_search_0.2_4096_1e-05_", folds=[0,2])
    # visualize_validation(run="param_search_0.2_4096_1e-05_", folds=[1,2])
    # life_grp, seqs, true_lbls, pred_lbls = extract_seq_group_for_predicted_aa_lbls(filename="w_lg_w_glbl_lbl_100ep.bin")
    # sp_pred_accs = get_pred_accs_sp_vs_nosp(life_grp, seqs, true_lbls, pred_lbls,v=True)
    # all_recalls, all_precisions, total_positives, false_positives, predictions = get_cs_acc(life_grp, seqs, true_lbls, pred_lbls)
