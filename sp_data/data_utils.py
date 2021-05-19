import torch
from torch.utils.data import Dataset, DataLoader
import random
import pickle
import pandas as pd
import os



class SPbinaryData:
    def __init__(self, threshold_pos=0.9, threshold_neg=0.15):
        self.data_folder = self.get_data_folder()
        self.threshold_pos = threshold_pos
        self.threshold_neg = threshold_neg
        self.train_datasets_per_fold = None
        self.test_datasets_per_fold = None
        if not os.path.exists(self.data_folder+"raw_seq_data_{}_{}.bin".format(threshold_pos, threshold_neg)):
            print("Did not find {}. Extracting...".format("raw_seq_data_{}_{}.bin".format(threshold_pos, threshold_neg)))
            sequences_data, labels = self.extract_labeled_data(threshold_pos=threshold_pos,
                                                               threshold_neg=threshold_neg)
            sequences_data, labels = self.shuffle(sequences_data, labels)
            pickle.dump([sequences_data, labels], open(self.data_folder+"raw_seq_data_{}_{}.bin".
                                                       format(threshold_pos,threshold_neg),'wb'))
        if not os.path.exists(self.data_folder+"bert_seq_data_{}_{}_1.bin".format(threshold_pos,threshold_neg)):
            print("Did not find bert-embeddings for the raw SP data. Please manually extract with bert_extraction "
                  "script using the data file {}".format("raw_seq_data_{}_{}.bin".format(threshold_pos,threshold_neg)))
            exit(1)
        self.form_cv_indices()

    def form_cv_indices(self):
        files = os.listdir(self.data_folder)
        all_emb_files = []
        for f in files:
            if "bert_seq_data_{}_{}_".format(self.threshold_pos, self.threshold_neg) in f:
                all_emb_files.append(f)
        num_files = len(all_emb_files)
        base_f_name = "bert_seq_data_{}_{}_".format(self.threshold_pos, self.threshold_neg)
        test_ds_inds = random.sample(list(range(num_files)), num_files)
        num_test_ds_per_fold = num_files//5

        test_datasets_per_fold, train_datasets_per_fold = [], []
        # if num_test_ds_per_fold * 5 < num_files:
        left_out_ds = num_files - num_test_ds_per_fold * 5
        for i in range(5):
            test_ds_current_fold_inds = test_ds_inds[i * num_test_ds_per_fold: (i+1) * num_test_ds_per_fold]
            if left_out_ds > 0:
                test_ds_current_fold_inds.append(test_ds_inds[-left_out_ds])
                left_out_ds -= 1
            train_ds_current_fold_inds = list(set(list(range(num_files)))- set(test_ds_current_fold_inds))
            train_datasets_per_fold.append([all_emb_files[i] for i in train_ds_current_fold_inds])
            test_datasets_per_fold.append([all_emb_files[i] for i in test_ds_current_fold_inds])
        self.train_datasets_per_fold = train_datasets_per_fold
        self.test_datasets_per_fold = test_datasets_per_fold

    def shuffle(self,sequences_data, labels):
        inds = random.sample(list(range(len(sequences_data))), len(sequences_data))
        sequences_data = [sequences_data[i] for i in inds]
        labels = [labels[i] for i in inds]
        return sequences_data, labels

    def parse_sequence_data(self,path="N_sequences.txt"):
        seq_data = pd.read_csv(path, sep='\t')
        names, seqs = seq_data.ID, seq_data.DNA_seq
        name2seq = {}
        for n, s in zip(names, seqs):
            name2seq[n] = s
        return name2seq

    def parse_label_data(self,path="Shrink_DA_result.txt"):
        name2density = {}
        label_data = pd.read_csv(path, sep="\t")
        for name, pos_dens in zip(label_data['Seq.names'], label_data['R pos.density']):
            name2density[name] = pos_dens
        return name2density

    def extract_labeled_data(self,threshold_pos=0.999, threshold_neg=0.001):
        name2seq, name2density = self.parse_sequence_data(path=self.data_folder+"N_sequences.txt"), \
                                 self.parse_label_data(path=self.data_folder+"Shrink_DA_result.txt")
        sequence_data, labels = [], []
        for name, seq in name2seq.items():
            if name2density[name] < threshold_neg:
                sequence_data.append(seq)
                labels.append(0)
            elif name2density[name] > threshold_pos:
                sequence_data.append(seq)
                labels.append(1)
        return sequence_data, labels


    def get_data_folder(self):
        if os.path.exists("/scratch/work/dumitra1"):
            return "/scratch/work/dumitra1/sp_data/"
        elif os.path.exists("/home/alex"):
            return "sp_data/"
        else:
            return "/scratch/project2003818/dumitra1/sp_data/"



class BinarySPDataset(Dataset):
    def __init__(self, data_file_path):
        data = pickle.load(open(data_file_path, "rb"))
        tcrs, embs_lbls = list(data.keys()), list(data.values())
        self.data = []
        for t,el in zip(tcrs, embs_lbls):
            self.data.append([t,el[0],el[1]])
        del data
        self.data = pd.DataFrame(self.data, columns=["sequence", "embeddings", "label"])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data.iloc[idx, 0]
        emb = self.data.iloc[idx, 1]
        lbl = self.data.iloc[idx, 2]
        sample = {'seq': seq,'emb':emb,'lbl':lbl}
        return sample
