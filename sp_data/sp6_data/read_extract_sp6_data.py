import pickle
import random
from Bio import SeqIO

ind = 0

def create_labeled_by_sp6_partition(all_ids, all_seqs, all_lbls):
    partition_2_info = {}
    for i, s, l in zip(all_ids, all_seqs, all_lbls):
        partition = int(i.split("|")[-1])
        if partition in partition_2_info:
            partition_2_info[partition][0].append(str(s))
            partition_2_info[partition][1].append(str(l))
            partition_2_info[partition][2].append(i)
        else:
            partition_2_info[partition] = [[],[],[]]
            partition_2_info[partition][0] = [str(s)]
            partition_2_info[partition][1] = [str(l)]
            partition_2_info[partition][2] = [i]
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

for seq_record in SeqIO.parse("train_set.fasta", "fasta"):
    seqs.append(seq_record.seq[:len(seq_record.seq) // 2])
    lbls.append(seq_record.seq[len(seq_record.seq) // 2:])
    ids.append(seq_record.id)


# for seq_record in SeqIO.parse("benchmark_set_sp5.fasta", "fasta"):
#     id_sequences_train.append((seq_record.id, seq_record.seq))
#     cat = get_cat(str(seq_record.seq))
#     if cat in train_categories2count:
#         train_categories2count[cat] += 1
#     elif cat not in train_categories2count:
#         train_categories2count[cat] = 1



partition_2_info = create_labeled_by_sp6_partition(ids, seqs, lbls)
for part_no, info in partition_2_info.items():
    pickle.dump(info, open("sp6_partitioned_data_{}.bin".format(part_no), "wb"))
