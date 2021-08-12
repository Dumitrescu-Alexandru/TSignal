import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import random
import pickle
import pandas as pd
import os


class SPCSpredictionData:
    def __init__(self, lbl2ind=None):
        self.lbl2ind = {}
        self.data_folder = self.get_data_folder()
        self.lg2ind = {}
        if lbl2ind is None:
            self.form_lbl_inds()
        else:
            self.lbl2ind = lbl2ind

    def form_lbl_inds(self):
        parts = [0, 1, 2]
        all_unique_lbls = set()
        for p in parts:
            part_dict = pickle.load(open(self.data_folder + "sp6_partitioned_data_{}_0.bin".format(p), "rb"))
            for (_, lbls, _) in part_dict.values():
                all_unique_lbls.update(lbls)
        self.lbl2ind = {l: ind for ind, l in enumerate(all_unique_lbls)}
        # add special tokens at the end of the dictionary
        unique_tkns = len(self.lbl2ind.keys())
        PAD_IDX, BOS_IDX, EOS_IDX = unique_tkns, unique_tkns + 1, unique_tkns + 2
        self.lbl2ind["PD"] = PAD_IDX
        self.lbl2ind["BS"] = BOS_IDX
        self.lbl2ind["ES"] = EOS_IDX

        all_unique_lgs = set()
        for p in parts:
            part_dict = pickle.load(open(self.data_folder + "sp6_partitioned_data_{}_0.bin".format(p), "rb"))
            for (_, _, lg) in part_dict.values():
                all_unique_lgs.add(lg)
        self.lg2ind = {l: ind for ind, l in enumerate(all_unique_lgs)}

    def get_data_folder(self):
        if os.path.exists("/scratch/work/dumitra1"):
            return "/scratch/work/dumitra1/sp_data/"
        elif os.path.exists("/home/alex"):
            return "sp_data/"
        else:
            return "/scratch/project2003818/dumitra1/sp_data/"


class CSPredsDataset(Dataset):
    def __init__(self, lbl2inds, partitions, data_folder):
        self.life_grp, self.seqs, self.lbls = [], [], []
        for p in partitions:
            data_dict = pickle.load(open(data_folder + "sp6_partitioned_data_{}_0.bin".format(p), "rb"))
            self.seqs.extend(list(data_dict.keys()))
            self.lbls.extend([[lbl2inds[l] for l in label] for (_, label, _) in data_dict.values()])
            self.life_grp.extend([life_grp for (_, _, life_grp) in data_dict.values()])

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, item):
        return {"seq": self.seqs[item], "lbl": self.lbls[item], "lg":self.life_grp[item]}


class SPbinaryData:
    def __init__(self, threshold_pos=0.9, threshold_neg=0.15, limit_seq=100, data="mammal"):
        self.data_folder = self.get_data_folder()
        self.threshold_pos = threshold_pos
        self.threshold_neg = threshold_neg
        self.train_datasets_per_fold = None
        self.test_datasets_per_fold = None
        self.data = data
        if data == "sp6data":
            if not os.path.exists(self.data_folder + "raw_sp6_train_data_0.bin"):
                print("Bert embeddings for sp6 data not found. Please extract the raw sequence files with "
                      "read_extract_sp6_data.py and then accordingly tun bert_extraction.py")
                exit(1)
        elif data == "mammal":
            if not os.path.exists(self.data_folder + "raw_seq_data_{}_{}.bin".format(threshold_pos, threshold_neg)):
                print("Did not find {}. Extracting...".format(
                    "raw_seq_data_{}_{}.bin".format(threshold_pos, threshold_neg)))
                sequences_data, labels, names = self.extract_labeled_data(threshold_pos=threshold_pos,
                                                                          threshold_neg=threshold_neg,
                                                                          limit_seq=limit_seq)
                sequences_data, labels, names = self.shuffle_w_even_lbl_no(sequences_data, labels, names)
                pickle.dump([sequences_data, labels, names], open(self.data_folder + "raw_seq_data_{}_{}.bin".
                                                                  format(threshold_pos, threshold_neg), 'wb'))
            if not os.path.exists(self.data_folder + "bert_seq_data_{}_{}_1.bin".format(threshold_pos, threshold_neg)):
                print("Did not find bert-embeddings for the raw SP data. Please manually extract with bert_extraction "
                      "script using the data file {}".format(
                    "raw_seq_data_{}_{}.bin".format(threshold_pos, threshold_neg)))
                exit(1)
        self.form_cv_indices()

    def form_cv_indices(self):
        np.random.seed(123)
        data_file_header = "_sp6" if self.data == "sp6data" else ""
        sp6data = self.data == "sp6data"
        if os.path.exists(self.data_folder + "train_datasets_per_fold{}.bin".format(data_file_header)):
            self.train_datasets_per_fold = pickle.load(open(self.data_folder +
                                                            "train_datasets_per_fold{}.bin".format(data_file_header),
                                                            "rb"))
            self.test_datasets_per_fold = pickle.load(open(self.data_folder +
                                                           "test_datasets_per_fold{}.bin".format(data_file_header),
                                                           "rb"))
        else:
            files = os.listdir(self.data_folder)
            all_emb_files = []
            for f in files:
                if "bert_seq_data_{}_{}_".format(self.threshold_pos, self.threshold_neg) in f and not sp6data:
                    all_emb_files.append(f)
                elif "raw_sp6_train_data_" in f and sp6data:
                    all_emb_files.append(f)

            num_files = len(all_emb_files)
            test_ds_inds = np.random.choice(list(range(num_files)), num_files, replace=False)
            num_test_ds_per_fold = num_files // 5
            test_datasets_per_fold, train_datasets_per_fold = [], []
            # if num_test_ds_per_fold * 5 < num_files:
            left_out_ds = num_files - num_test_ds_per_fold * 5
            for i in range(5):
                test_ds_current_fold_inds = list(test_ds_inds[i * num_test_ds_per_fold: (i + 1) * num_test_ds_per_fold])
                if left_out_ds > 0:
                    test_ds_current_fold_inds.append(test_ds_inds[-left_out_ds])
                    left_out_ds -= 1
                train_ds_current_fold_inds = list(set(list(range(num_files))) - set(test_ds_current_fold_inds))
                train_datasets_per_fold.append([all_emb_files[i] for i in train_ds_current_fold_inds])
                test_datasets_per_fold.append([all_emb_files[i] for i in test_ds_current_fold_inds])
            self.train_datasets_per_fold = train_datasets_per_fold
            self.test_datasets_per_fold = test_datasets_per_fold
            pickle.dump(self.train_datasets_per_fold, open(self.data_folder +
                                                           "train_datasets_per_fold{}.bin".format(data_file_header),
                                                           "wb"))
            pickle.dump(self.test_datasets_per_fold, open(self.data_folder +
                                                          "test_datasets_per_fold{}.bin".format(data_file_header),
                                                          "wb"))

    def shuffle_w_even_lbl_no(self, sequences_data, labels, names, no_sequences=4500):
        """
        :parameter:
            sequences_data: list(str). List of sequences that should have the N-terminus + aa from the mature protein
            labels: list(int): list of labels 0/1 (is/is not SP)
            names: list(str): list of name indicators for each protein sequence
            no_sequences: int: should match the length of the dataset-chunks that bert_extraction will then use
        :return
            lists sequences_data, labels, names with shuffled data s.t. sections of data
            [i*no_sequences:(i+1)*no_sequences] will have the same number of positive and negative labels
        """
        lbl_array = np.array(labels)
        negative_indices, positive_indices = np.where(lbl_array == 0)[0].reshape(-1), \
                                             np.where(lbl_array == 1)[0].reshape(-1)
        pos_sample_ratio, neg_sample_ratio = len(positive_indices) / len(lbl_array), \
                                             len(negative_indices) / len(lbl_array)
        not_sampled_pos, not_sampled_neg = set(positive_indices), set(negative_indices)
        all_shuffled_inds = []
        current_all_samples = []
        while len(not_sampled_neg) != 0:
            current_all_samples = []
            if len(not_sampled_neg) + len(not_sampled_pos) < no_sequences:
                # add remaining samples to the last datasets chunk
                current_all_samples.extend(not_sampled_pos)
                current_all_samples.extend(not_sampled_neg)
                # shuffle
                current_all_samples = random.sample(current_all_samples, len(current_all_samples))
                all_shuffled_inds.extend(current_all_samples)
                not_sampled_pos, not_sampled_neg = set(), set()

            else:
                # extract the same pos-neg ratio of negatives per dataset-chunk
                current_pos_samples, \
                current_neg_samples = random.sample(not_sampled_pos, int(no_sequences * pos_sample_ratio)), \
                                      random.sample(not_sampled_neg, int(no_sequences * neg_sample_ratio) + 1)
                current_all_samples.extend(current_pos_samples)
                current_all_samples.extend(current_neg_samples)
                not_sampled_pos, not_sampled_neg = not_sampled_pos - set(current_pos_samples), \
                                                   not_sampled_neg - set(current_neg_samples)
                # shuffle
                current_all_samples = random.sample(current_all_samples, len(current_all_samples))
                all_shuffled_inds.extend(current_all_samples)
        sequences_data, labels, names = [sequences_data[i] for i in all_shuffled_inds], \
                                        [labels[i] for i in all_shuffled_inds], \
                                        [names[i] for i in all_shuffled_inds]
        print(len(set(sequences_data)), len(sequences_data))
        return sequences_data, labels, names

    def shuffle(self, sequences_data, labels, names):
        inds = random.sample(list(range(len(sequences_data))), len(sequences_data))
        sequences_data = [sequences_data[i] for i in inds]
        labels = [labels[i] for i in inds]
        names = [names[i] for i in inds]
        return sequences_data, labels, names

    def parse_sequence_data(self, path="N_sequences.txt", limit_seq=-1):
        rest_of_seq = "TRATADAQSRMQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPP" \
                      "DQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGVVGKLGRQDPPVATR"
        seq_data = pd.read_csv(path, sep='\t')
        names, seqs = seq_data.ID, seq_data.AA_seq
        name2seq = {}
        for n, s in zip(names, seqs):
            if limit_seq != -1:
                aa_to_add = limit_seq - len(s)
                name2seq[n] = s + rest_of_seq[:aa_to_add]
            else:
                name2seq[n] = s + rest_of_seq
        return name2seq

    def parse_label_data(self, path="Shrink_DA_result.txt"):
        name2density = {}
        label_data = pd.read_csv(path, sep="\t")
        for name, pos_dens in zip(label_data['Seq.names'], label_data['R pos.density']):
            name2density[name] = pos_dens
        return name2density

    def extract_labeled_data(self, threshold_pos=0.999, threshold_neg=0.001, limit_seq=70, mature_protein_end=""):
        name2seq, name2density = self.parse_sequence_data(path=self.data_folder + "N_sequences.txt",
                                                          limit_seq=limit_seq), \
                                 self.parse_label_data(path=self.data_folder + "Shrink_DA_result.txt")
        sequence_data, labels, names = [], [], []

        for name, seq in name2seq.items():
            if name2density[name] < threshold_neg:
                sequence_data.append(seq)
                labels.append(0)
                names.append(name)
            elif name2density[name] > threshold_pos:
                sequence_data.append(seq)
                labels.append(1)
                names.append(name)
        return sequence_data, labels, names

    def get_data_folder(self):
        if os.path.exists("/scratch/work/dumitra1"):
            return "/scratch/work/dumitra1/sp_data/"
        elif os.path.exists("/home/alex"):
            return "sp_data/"
        else:
            return "/scratch/project2003818/dumitra1/sp_data/"


def collate_fn(batch):
    src_batch, tgt_batch, life_grp = [], [], []
    for sample in batch:
        src_batch.append(sample['seq'])
        tgt_batch.append(sample['lbl'])
        life_grp.append(sample['lg'])
    return src_batch, tgt_batch, life_grp


class BinarySPDataset(Dataset):
    def __init__(self, data_file_path, use_aa_len=70):
        data = pickle.load(open(data_file_path, "rb"))
        tcrs, embs_lbls = list(data.keys()), list(data.values())
        self.data = []
        for t, el in zip(tcrs, embs_lbls):
            self.data.append([t, el[0], el[1]])
        del data
        self.data = pd.DataFrame(self.data, columns=["sequence", "embeddings", "label"])
        self.use_aa_len = use_aa_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data.iloc[idx, 0]
        emb = self.data.iloc[idx, 1]

        if emb.shape[0] != self.use_aa_len:
            emb_ = np.zeros((self.use_aa_len, 1024))
            emb_[:emb.shape[0], :] = emb
            emb = emb_
        lbl = self.data.iloc[idx, 2]
        sample = {'seq': seq, 'emb': emb, 'lbl': lbl}
        return sample
