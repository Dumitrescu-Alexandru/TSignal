import matplotlib.pyplot as plt
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

    def get_acc_for_tolerence(ind, t_lbl):
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
    count_tol_fn, count_complete_fn, count_otherSPpred = 0, 0, 0
    for l, s, t, p in zip(life_grp, seqs, true_lbls, pred_lbls):
        life_grp, sp_info = l.split("|")
        ind = 0
        if sp_info == "SP":
            while p[ind] == "S" and ind < len(p) - 1:
                ind += 1
            if ind == 0 and life_grp == "POSITIVE" and v and ('L' not in p and 'T' not in p and 'P' not in p and 'M' not in p):
                count_complete_fn += 1
                print(p)
                print(t)
                print(s)
                print("\n")
                count_tol_fn += 1
            elif ind == 0 and life_grp == "POSITIVE" and v and ('L' in p or 'T' in p  or 'P' in p or 'M' in p):
                count_otherSPpred += 1
            elif get_acc_for_tolerence(ind, t)[3] == 0 and v and life_grp == "POSITIVE":
                count += 1
                print(p)
                print(t)
                print(s)
                print("\n")
                count_tol_fn += 1
            predictions[grp2_ind[life_grp]] += get_acc_for_tolerence(ind, t)

        elif sp_info != "SP" and p[ind] == "S":
            # count false positive predictions
            # if v and life_grp=="ARCHAEA":
            #     print(life_grp)
            #     print(p)
            #     print(t)
            #     print(s)
            #     print("\n")
            predictions[grp2_ind[life_grp]] += np.array([0, 0, 0, 0, 0, 1])
    if v:
        print(" count_tol_fn, count_complete_fn, count_otherSPpred",  count_tol_fn, count_complete_fn, count_otherSPpred)
    all_recalls = []
    all_precisions = []
    total_positives = []
    false_positives = []
    for life_grp, ind in grp2_ind.items():
        current_preds = predictions[grp2_ind[life_grp]]
        if v:
            print("Recall {}: {}".format(life_grp, [current_preds[i] / current_preds[-2] for i in range(len(current_preds) - 2)]))
            print("Prec {}: {}".format(life_grp, [current_preds[i] / (current_preds[i] +current_preds[-1]) for i in range(len(current_preds) - 2)]))
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
    all_recalls, all_precisions, total_positives, false_positives, predictions = get_cs_acc(life_grp, seqs, true_lbls, pred_lbls, v=v)
    return sp_pred_accs, all_recalls, all_precisions, total_positives, false_positives, predictions

def get_summary_sp_acc(sp_pred_accs):
    return np.mean(sp_pred_accs), sp_pred_accs[0]

def get_summary_cs_acc(all_cs_preds):
    return np.mean(np.array(all_cs_preds)), np.mean(all_cs_preds[0]), all_cs_preds[0][0]

def plot_losses(losses, name="param_search_0.2_2048_0.0001_"):
    train_loss, valid_loss = losses
    fig, axs = plt.subplots(1,1,figsize=(12,8))

    axs.set_title("Train and validation loss over epochs")
    axs.plot(train_loss, label="Train loss")
    axs.plot(valid_loss, label="Validation loss")
    axs.set_xlabel("Epochs")
    axs.set_ylabel("Loss")
    axs.legend()
    axs.set_ylim(0,0.2)
    plt.savefig("/home/alex/Desktop/sp6_ds_transformer_nmt_results/" + name+"loss.png")

def plot_mcc(mccs, name="param_search_0.2_2048_0.0001_"):
    euk_mcc, neg_mcc, pos_mcc, arc_mcc = mccs
    fig, axs = plt.subplots(2,2,figsize=(12,8))
    axs[0,0].plot(euk_mcc, label="Eukaryote mcc")
    axs[0,0].set_ylabel("mcc")
    axs[0,0].set_ylim(-1.1, 1.1)
    axs[0,0].legend()

    axs[0, 1].plot(neg_mcc, label="Negative mcc")
    axs[0, 1].set_ylabel("mcc")
    axs[0, 1].set_ylim(-1.1, 1.1)
    axs[0, 1].legend()

    axs[1, 0].plot(pos_mcc, label="Positive mcc")
    axs[1, 0].legend()
    axs[1, 0].set_ylim(-1.1, 1.1)
    axs[1, 0].set_xlabel("epochs")

    axs[1, 1].plot(arc_mcc, label="Archaea mcc")
    axs[1, 1].legend()
    axs[1, 1].set_ylim(-1.1, 1.1)
    axs[1, 1].set_xlabel("epochs")

    plt.savefig("/home/alex/Desktop/sp6_ds_transformer_nmt_results/{}_{}.png".format(name, "mcc"))

def extract_and_plot_prec_recall(results, metric="recall", name="param_search_0.2_2048_0.0001_"):
    cs_res_euk, cs_res_neg, cs_res_pos, cs_res_arc = results
    fig, axs = plt.subplots(2,2,figsize=(12,8))
    for i in range(4):
        axs[0,0].plot(cs_res_euk[i], label="Eukaryote {} tol={}".format(metric, i))
        axs[0,0].set_ylabel(metric)
        axs[0,0].legend()
        axs[0,0].set_ylim(-0.1, 1.1)

        axs[0, 1].plot(cs_res_neg[i], label="Negative {} tol={}".format(metric, i))
        axs[0, 1].set_ylabel(metric)
        axs[0, 1].legend()
        axs[0, 1].set_ylim(-0.1, 1.1)

        axs[1, 0].plot(cs_res_pos[i], label="Positive {} tol={}".format(metric, i))
        axs[1, 0].legend()
        axs[1, 0].set_xlabel("epochs")
        axs[1, 0].set_ylim(-0.1, 1.1)

        axs[1, 1].plot(cs_res_arc[i], label="Archaea {} tol={}".format(metric, i))
        axs[1, 1].legend()
        axs[1, 1].set_xlabel("epochs")
        axs[1, 1].set_ylim(-0.1, 1.1)

    plt.savefig("/home/alex/Desktop/sp6_ds_transformer_nmt_results/{}_{}.png".format(name, metric))

def visualize_validation(run="param_search_0.2_2048_0.0001_", folds=[0,1]):
    all_results = []
    euk_mcc, neg_mcc, pos_mcc, arc_mcc, train_loss, valid_loss, cs_recalls_euk, cs_recalls_neg, cs_recalls_pos, \
        cs_recalls_arc, cs_precs_euk, cs_precs_neg, cs_precs_pos, cs_precs_arc = extract_results(run, folds=folds)
    plot_mcc([euk_mcc, neg_mcc, pos_mcc, arc_mcc], name=run)
    plot_losses([train_loss, valid_loss], name=run)
    extract_and_plot_prec_recall([cs_recalls_euk, cs_recalls_neg, cs_recalls_pos, cs_recalls_arc], metric="recall", name=run)
    extract_and_plot_prec_recall([cs_precs_euk, cs_precs_neg, cs_precs_pos, cs_precs_arc], metric="precision", name=run)
    # extract_and_plot_losses(lines)


def extract_results(run="param_search_0.2_2048_0.0001_", folds=[0, 1], folder='results_param_s_2/'):
    euk_mcc, neg_mcc, pos_mcc, arc_mcc = [], [], [] ,[]
    train_loss, valid_loss = [], []
    cs_recalls_euk, cs_recalls_neg, cs_recalls_pos, cs_recalls_arc = [[], [], [], []], [[], [], [], []], \
                                                                                 [[], [], [], []], [[], [], [], []]
    cs_precs_euk, cs_precs_neg, cs_precs_pos, cs_precs_arc = [[], [], [], []], [[], [], [], []], \
                                                                         [[], [], [], []], [[], [], [], []]
    with open(folder + run + "{}_{}.log".format(folds[0], folds[1]), "rt") as f:
        lines = f.readlines()
    for l in lines:
        if "sp_pred mcc" in l and "VALIDATION" in l:
            mccs = l.split(":")[-1].replace(" ","").split(",")
            euk_mcc.append(float(mccs[0]))
            neg_mcc.append(float(mccs[1]))
            pos_mcc.append(float(mccs[2]))
            arc_mcc.append(float(mccs[3]))
        elif "train/validation" in l:
            train_l, valid_l = l.split(":")[-1].replace(" ","").split("/")
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

    return euk_mcc, neg_mcc, pos_mcc, arc_mcc, train_loss, valid_loss, cs_recalls_euk, cs_recalls_neg, cs_recalls_pos,\
           cs_recalls_arc, cs_precs_euk, cs_precs_neg, cs_precs_pos, cs_precs_arc

def extract_mean_test_results(run="param_search_0.2_2048_0.0001", result_folder="results_param_s_2/"):
    full_dict_results = {}
    epochs = []
    for tr_folds in [[0,1],[1,2],[0,2]]:
        with open(result_folder+run+ "_{}_{}.log".format(tr_folds[0], tr_folds[1]), "rt") as f:
            lines = f.readlines()
            epochs.append(int(lines[-2].split(" ")[2]))
    avg_epoch = np.mean(epochs)
    print("Results found on epochs: {}, {}, {}".format(*epochs))

    for tr_folds in [[0,1],[1,2],[0,2]]:
        res_dict = pickle.load(open(result_folder + run + "_{}_{}_best.bin".format(tr_folds[0], tr_folds[1]), "rb"))
        full_dict_results.update(res_dict)
    life_grp, seqs, true_lbls, pred_lbls = extract_seq_group_for_predicted_aa_lbls(filename="w_lg_w_glbl_lbl_100ep.bin", dict_=full_dict_results)
    mccs = get_pred_accs_sp_vs_nosp(life_grp, seqs, true_lbls, pred_lbls,v=False)
    if "param_search_w_nl_nh_0.0_4096_1e-05_4_4" in run:
        v = True
    else:
        v=False
    all_recalls, all_precisions, _, _, _ = \
        get_cs_acc(life_grp, seqs, true_lbls, pred_lbls, v=v)
    return mccs, all_recalls, all_precisions, avg_epoch

def get_best_corresponding_eval_mcc(result_folder="results_param_s_2/", model=""):
    tr_fold = [[0,1],[1,2],[0,2]]
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







def extract_all_param_results(result_folder="results_param_s_2/"):
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
        mccs, all_recalls, all_precisions, avg_epoch= extract_mean_test_results(run=u_p, result_folder=result_folder)
        mdl2results[ind] = (mccs, list(np.reshape(np.array(all_recalls),-1)),
                            list(np.reshape(np.array(all_precisions),-1)), avg_epoch)
        mdlind2mdlparams[ind] = u_p
        eukaryote_mcc.append(get_best_corresponding_eval_mcc(result_folder, u_p))
    best_to_worst_mdls = np.argsort(eukaryote_mcc)[::-1]
    print("\n\nMCC TABLE\n\n")
    for mdl_ind in best_to_worst_mdls:
        print(" & ".join(mdlind2mdlparams[mdl_ind].split("_")[6:]), "&", " & ".join([str(round(mcc,3))
                     for mcc in mdl2results[mdl_ind][0]]), "&", round(mdl2results[mdl_ind][-1],3), "\\\\ \\hline")

    print("\n\nRecall table \n\n")
    for mdl_ind in best_to_worst_mdls:
        print(" & ".join(mdlind2mdlparams[mdl_ind].split("_")[6:]), "&",
              " & ".join([str(round(rec,3)) for rec in mdl2results[mdl_ind][1]]), "&", round(mdl2results[mdl_ind][-1],3), "\\\\ \\hline")

    print("\n\nPrec table \n\n")
    for mdl_ind in best_to_worst_mdls:
        print(" & ".join(mdlind2mdlparams[mdl_ind].split("_")[6:]), "&",
              " & ".join([str(round(rec,3)) for rec in mdl2results[mdl_ind][2]]), "&", round(mdl2results[mdl_ind][-1],3), "\\\\ \\hline")


if __name__ == "__main__":
    # extract_all_param_results()
    # extract_mean_test_results(run="param_search_0_2048_1e-05")
    visualize_validation(run="param_search_w_nl_nh_0.0_4096_1e-05_4_4_", folds=[0,1])
    # visualize_validation(run="param_search_0.2_4096_1e-05_", folds=[0,2])
    # visualize_validation(run="param_search_0.2_4096_1e-05_", folds=[1,2])
    # life_grp, seqs, true_lbls, pred_lbls = extract_seq_group_for_predicted_aa_lbls(filename="w_lg_w_glbl_lbl_100ep.bin")
    # sp_pred_accs = get_pred_accs_sp_vs_nosp(life_grp, seqs, true_lbls, pred_lbls,v=True)
    # all_recalls, all_precisions, total_positives, false_positives, predictions = get_cs_acc(life_grp, seqs, true_lbls, pred_lbls)

