import pickle
import random

import matplotlib.pyplot as plt
from Bio import SeqIO

def split_train_test_partitions(partition, split_perc=0.1):
    lgrp_and_sptype_2_inds = {}
    for ind, (seq, lbls, glbl_info) in enumerate(zip(*partition)):
        life_grp_and_sp_type = "_".join(glbl_info.split("|")[1:3])
        if life_grp_and_sp_type in lgrp_and_sptype_2_inds:
            lgrp_and_sptype_2_inds[life_grp_and_sp_type].append(ind)
        else:
            lgrp_and_sptype_2_inds[life_grp_and_sp_type] = [ind]
    seq_test, seq_train, lbls_test, lbls_train, glbl_info_train, glbl_info_test = [],[],[],[],[],[]
    for lgrp_and_sp_type, inds in lgrp_and_sptype_2_inds.items():
        if int(len(inds) * split_perc) > 0:
            no_of_test_items = int(len(inds) * split_perc)
            test_inds = random.sample(inds, no_of_test_items)
            train_inds = set(inds) - set(test_inds)

            seq_train.extend([partition[0][tri] for tri in train_inds])
            lbls_train.extend([partition[1][tri] for tri in train_inds])
            glbl_info_train.extend([partition[2][tri] for tri in train_inds])
            seq_test.extend([partition[0][tsi] for tsi in test_inds])
            lbls_test.extend([partition[1][tsi] for tsi in test_inds])
            glbl_info_test.extend([partition[2][tsi] for tsi in test_inds])
        else:
            # if not enough samples for the life group with the specific global SP type, add samples only to the train
            # set of the partition
            seq_train.extend([partition[0][i] for i in inds])
            lbls_train.extend([partition[1][i] for i in inds])
            glbl_info_train.extend([partition[2][i] for i in inds])
    return [seq_train, lbls_train, glbl_info_train], [seq_test, lbls_test, glbl_info_test]


def create_labeled_by_sp6_partition(all_ids, all_seqs, all_lbls):
    count_types = {}
    type2someseq = {}
    partition_2_info = {}
    for i, s, l in zip(all_ids, all_seqs, all_lbls):

        life_grp, partition = i.split("|")[1], int(i.split("|")[-1])
        if i.split("|")[2] in count_types:
            count_types[i.split("|")[2]] += 1
        else:
            count_types[i.split("|")[2]] = 1
        if i.split("|")[2] not in type2someseq:
            type2someseq[i.split("|")[2]] = l

        if partition in partition_2_info:
            partition_2_info[partition][0].append(str(s))
            partition_2_info[partition][1].append(str(l))
            partition_2_info[partition][2].append(i)
        else:
            partition_2_info[partition] = [[],[],[]]
            partition_2_info[partition][0] = [str(s)]
            partition_2_info[partition][1] = [str(l)]
            partition_2_info[partition][2] = [i]
    train_partitions, test_partitions = {}, {}
    for part, info in partition_2_info.items():
        train_current_part, test_current_part = split_train_test_partitions(info)
        train_partitions[part] = train_current_part
        test_partitions[part] = test_current_part
    return partition_2_info

def create_labeled_sp6_seqs(id_and_seqs):
    ids, seqs, lbls = [], [], []
    for id_and_seq in id_and_seqs:
        seq_len = len(id_and_seq[1]) // 2
        seq_id, seq, lbl = id_and_seq[0], str(id_and_seq[1][:seq_len]), str(id_and_seq[1][seq_len:])
        lbl = 1 if ("P" in lbl or "T" in lbl or "S" in lbl or "L" in lbl) else 0
        ids.append(seq_id)
        seqs.append(seq)
        lbls.append(lbl)
    return ids, seqs, lbls


def create_files(inds, lbls, seqs, train=False):
    unique_seqs = set()
    # There are some duplicate seqs somehow. Remove them.
    inds_, lbls_, seqs_ = [], [], []
    for i,l,s in zip(inds, lbls, seqs):
        if s not in unique_seqs:
            unique_seqs.add(s)
            inds_.append(i)
            lbls_.append(l)
            seqs_.append(s)
    inds, lbls, seqs = inds_, lbls_, seqs_
    file_size = 4100
    if train:
        lbl2inds_seqs = {0: [], 1: []}
        for ind, l in enumerate(lbls):
            lbl2inds_seqs[l].append(ind)
        neg_ratio = len(lbl2inds_seqs[0]) / (len(lbl2inds_seqs[1]) + len(lbl2inds_seqs[0]))
        pos_ratio = len(lbl2inds_seqs[1]) / (len(lbl2inds_seqs[1]) + len(lbl2inds_seqs[0]))
        neg_inds, pos_inds = set(lbl2inds_seqs[0]), set(lbl2inds_seqs[1])
        datasets = []
        while neg_inds:
            current_ds_seqs = []
            current_ds_ids = []
            current_ds_lbls = []
            neg_inds_current_ds = random.sample(neg_inds, min(len(neg_inds), int(file_size * neg_ratio + 1)))
            pos_inds_current_ds = random.sample(pos_inds, min(len(pos_inds), int(file_size * pos_ratio)))
            pos_inds = pos_inds - set(pos_inds_current_ds)
            neg_inds = neg_inds - set(neg_inds_current_ds)
            current_ds_seqs.extend([seqs[i] for i in pos_inds_current_ds])
            current_ds_seqs.extend([seqs[i] for i in neg_inds_current_ds])
            current_ds_ids.extend([inds[i] for i in pos_inds_current_ds])
            current_ds_ids.extend([inds[i] for i in neg_inds_current_ds])
            current_ds_lbls.extend([lbls[i] for i in pos_inds_current_ds])
            current_ds_lbls.extend(lbls[i] for i in neg_inds_current_ds)
            datasets.append([current_ds_seqs, current_ds_lbls, current_ds_ids])
        all_lbls, all_seqs, all_ids = [], [], []
        for ds in datasets:
            all_seqs.extend(ds[0])
            all_lbls.extend(ds[1])
            all_ids.extend(ds[2])
        data = [all_seqs, all_lbls,all_ids]
        pickle.dump(data, open("raw_sp6_train_data.bin", "wb"))
    else:
        pickle.dump([seqs, lbls, inds], open("raw_sp6_bench_data.bin", "wb"))

seqs, lbls, ids, global_lbls = [], [], [], []
uid = []
for seq_record in SeqIO.parse("train_set.fasta", "fasta"):
    seqs.append(seq_record.seq[:len(seq_record.seq) // 2])
    lbls.append(seq_record.seq[len(seq_record.seq) // 2:])
    ids.append(seq_record.id)
    uid.append(seq_record.id.split("|")[-1])

# for seq_record in SeqIO.parse("benchmark_set_sp5.fasta", "fasta"):
#     id_sequences_train.append((seq_record.id, seq_record.seq))
#     cat = get_cat(str(seq_record.seq))
#     if cat in train_categories2count:
#         train_categories2count[cat] += 1
#     elif cat not in train_categories2count:
#         train_categories2count[cat] = 1
lgandsptype2count = {}
# for i,s,l in zip(ids, seqs, lbls):
#     if "_".join(i.split("|")[1:3]) not in lgandsptype2count:
#         lgandsptype2count["_".join(i.split("|")[1:3])] = 1
#     else:
#         lgandsptype2count["_".join(i.split("|")[1:3])] += 1

# print(lgandsptype2count)
# exit(1)
lgandsptype2counts = []
lgandsptype2count_total = {}
preds_best_mdl = {}
for tr_f in [[0,1],[0,2],[1,2]]:
    # best_mdl = "../../misc/huge_param_search/parameter_search_patience_30use_glbl_lbls_use_glbl_lbls_version_1_weight_0.1_lr_1e-05_nlayers_3_nhead_16_lrsched_none_trFlds_"
    best_mdl = "../../misc/detailed_sp/deailed_sp_v1_"
    best_mdl = best_mdl  + "{}_{}_best.bin".format(tr_f[0], tr_f[1])
    preds_best_mdl.update(pickle.load(open(best_mdl, "rb")))
print(len(set(preds_best_mdl.keys())))
import numpy as np


def get_hydrophobicity(seq, lbls, window=7, sp_type="S"):
    is_sp = lbls[0] == sp_type
    add_lr_aa = window//2
    start_pos, end_pos = add_lr_aa, len(seq)-add_lr_aa
    all_hydrophobs = []
    relative_diff = 0
    for i in range(start_pos, end_pos):
        total_hydro = 0
        for j in range(i - add_lr_aa, i + add_lr_aa):
            total_hydro += kyte_doolittle_hydrophobicity[seq[j]]
        all_hydrophobs.append(total_hydro)
    if is_sp:
        end_sp = lbls.rfind(sp_type)
        h_region_ind = np.argmax(all_hydrophobs[:end_sp - 3]) + 3
        relative_diff = end_sp - h_region_ind
        return relative_diff, h_region_ind, end_sp

kyte_doolittle_hydrophobicity = {"A":1.8, "C":2.5, "D":-3.5, "E":-3.5, "F":2.8, "G":-0.4, "H":-3.2, "I":4.5,"K":-3.9,
                                 "L":3.8, "M":1.9, "N":-3.5, "P":-1.6, "Q":-3.5, "R":-4.5, "S":-0.8, "T":-0.7, "V":4.2, "W":-0.9, "Y":-1.3}
partition_2_info = create_labeled_by_sp6_partition(ids, seqs, lbls)
for part_id, part in partition_2_info.items():
    lgandsptype2count = {}

    seqs, lbls, ids = part
    for i,s,l in zip(ids, seqs, lbls):
        if i.split("|")[-2] == "TAT":# and i.split("|")[1] == "EUKARYA" and preds_best_mdl[s][0] != "S":# and i.split("|")[1] == "NEGATIVE" and (preds_best_mdl[s][0] == "L" or preds_best_mdl[s][0] == "T"):
            print(get_hydrophobicity(s, l, sp_type="T"))
            print(s)
            print(l)
            print(preds_best_mdl[s])
        if "_".join(i.split("|")[1:3]) not in lgandsptype2count:
            lgandsptype2count["_".join(i.split("|")[1:3])] = 1
        if "_".join(i.split("|")[1:3]) not in lgandsptype2count_total:
            lgandsptype2count_total["_".join(i.split("|")[1:3])] = 1
        else:
            lgandsptype2count["_".join(i.split("|")[1:3])] += 1
            lgandsptype2count_total["_".join(i.split("|")[1:3])] += 1
    if "ARCHAEA_SP" not in lgandsptype2count:
        lgandsptype2count["ARCHAEA_SP"] = 0
    lgandsptype2counts.append(lgandsptype2count)
    print(part_id, lgandsptype2count)

plot_for = ["EUKARYA_NO_SP", "EUKARYA_SP", "NEGATIVE_NO_SP", "NEGATIVE_SP", "POSITIVE_NO_SP", "POSITIVE_SP", "ARCHAEA_NO_SP", "ARCHAEA_SP"]

# def plot_lg_dists(lgandsptype2counts):
#     import numpy as np
#     line_w = 0.25
#     x_positions = []
#     heights = []
#     totals = []
#     for i in range(1, 9):
#         x_positions.extend([i-line_w, i, i+line_w])
#     for pf in plot_for:
#         total = lgandsptype2count_total[pf]
#         totals.append(total)
#         heights.extend([lgandsptype2counts[0][pf] /total, lgandsptype2counts[1][pf] /total, lgandsptype2counts[2][pf] /total])
#     plt.bar([x_positions[i*3] for i in range(8)], [heights[i*3] for i in range(8)], width=line_w, label="partition 1")
#     plt.bar([x_positions[i*3+1] for i in range(8)], [heights[i*3+1] for i in range(8)], width=line_w, label="partition 2")
#     plt.bar([x_positions[i*3+2] for i in range(8)], [heights[i*3+2] for i in range(8)], width=line_w, label="partition 3")
#     plt.legend()
#     plt.xticks(list(range(1,9)),[plot_for[i] + "\n" + str(totals[i]) for i in range(8)])
#     plt.xlabel("Life group, SP/NO-SP; No. of datapoints")
#     plt.ylabel("Percentage from that life group")
#     plt.show()
#
# plot_lg_dists(lgandsptype2counts)

exit(1)



for part_no, info in partition_2_info.items():
    train_part_info, test_part_info = split_train_test_partitions(info)
    # the split is done evenely across all global labels in conjunction with the life group information
    pickle.dump(train_part_info, open("sp6_partitioned_data_train_{}.bin".format(part_no), "wb"))
    pickle.dump(test_part_info, open("sp6_partitioned_data_test_{}.bin".format(part_no), "wb"))
