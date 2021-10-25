import re
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import random
import pickle
import pandas as pd
import os
from Bio import SeqIO


class SPCSpredictionData:
    def __init__(self, lbl2ind=None, form_sp_reg_data=False, simplified=True, very_simplified=True):
        self.aa2ind = {}
        self.lbl2ind = {}
        self.glbl_lbl_2ind = {}
        self.simplified = simplified if not very_simplified else True
        self.very_simplified = very_simplified
        self.data_folder = self.get_data_folder()
        self.lg2ind = {}
        self.form_sp_reg_data = form_sp_reg_data
        if form_sp_reg_data:
            self.set_dicts(form_sp_reg_data)
            self.lbl2ind, self.lg2ind, self.glbl_lbl_2ind, self.aa2ind = pickle.load(
                open("sp6_dicts_subregion_lbls.bin", "rb"))
        else:
            self.lbl2ind, self.lg2ind, self.glbl_lbl_2ind, self.aa2ind = pickle.load(open("sp6_dicts.bin", "rb"))
        if not os.path.exists(self.get_data_folder() + "sp6_partitioned_data_sublbls_test_0.bin"):
            self.form_subregion_sp_data()
        # if os.path.exists("sp6_dicts.bin"):
        #     self.lbl2ind, self.lg2ind, self.glbl_lbl_2ind, self.aa2ind = pickle.load(open("sp6_dicts.bin", "rb"))
        # else:
        #     self.form_lbl_inds()

    def set_dicts(self, form_sp_reg_data=False):

        if form_sp_reg_data:
            if self.very_simplified:
                dicts = [{'S': 0, 'O': 1, 'M': 2, 'I': 3, 'PD': 4, 'BS': 5, 'ES': 6},
                         {'EUKARYA': 0, 'POSITIVE': 1, 'ARCHAEA': 2, 'NEGATIVE': 3},
                         {'NO_SP': 0, 'SP': 1, 'TATLIPO': 2, 'LIPO': 3, 'TAT': 4, 'PILIN': 5},
                         {'V': 0, 'R': 1, 'D': 2, 'E': 3, 'H': 4, 'A': 5, 'G': 6, 'Y': 7, 'W': 8, 'F': 9, 'M': 10,
                          'K': 11,
                          'L': 12,
                          'I': 13, 'C': 14, 'Q': 15, 'S': 16, 'P': 17, 'N': 18, 'T': 19, 'PD': 20, 'BS': 21,
                          'ES': 22}]
            elif self.simplified:
                dicts = [{'S': 0, 'R': 1, 'C': 2, 'O': 3, 'M': 4, 'I': 5, 'PD': 6, 'BS': 7, 'ES': 8},
                         {'EUKARYA': 0, 'POSITIVE': 1, 'ARCHAEA': 2, 'NEGATIVE': 3},
                         {'NO_SP': 0, 'SP': 1, 'TATLIPO': 2, 'LIPO': 3, 'TAT': 4, 'PILIN': 5},
                         {'V': 0, 'R': 1, 'D': 2, 'E': 3, 'H': 4, 'A': 5, 'G': 6, 'Y': 7, 'W': 8, 'F': 9, 'M': 10,
                          'K': 11,
                          'L': 12,
                          'I': 13, 'C': 14, 'Q': 15, 'S': 16, 'P': 17, 'N': 18, 'T': 19, 'PD': 20, 'BS': 21,
                          'ES': 22}]
            else:
                dicts = [
                    {'P': 0, 'n': 1, 'h': 2, 'c': 3, 'N': 4, 'H': 5, 'R': 6, 'C': 7, 'B': 8, 'O': 9, 'M': 10, 'I': 11,
                     'PD': 12, 'BS': 13, 'ES': 14},
                    {'EUKARYA': 0, 'POSITIVE': 1, 'ARCHAEA': 2, 'NEGATIVE': 3},
                    {'NO_SP': 0, 'SP': 1, 'TATLIPO': 2, 'LIPO': 3, 'TAT': 4, 'PILIN': 5},
                    {'V': 0, 'R': 1, 'D': 2, 'E': 3, 'H': 4, 'A': 5, 'G': 6, 'Y': 7, 'W': 8, 'F': 9, 'M': 10, 'K': 11,
                     'L': 12,
                     'I': 13, 'C': 14, 'Q': 15, 'S': 16, 'P': 17, 'N': 18, 'T': 19, 'PD': 20, 'BS': 21, 'ES': 22}]
        else:
            dicts = [{'P': 0, 'S': 1, 'O': 2, 'M': 3, 'L': 4, 'I': 5, 'T': 6, 'PD': 7, 'BS': 8, 'ES': 9},
                     {'EUKARYA': 0, 'POSITIVE': 1, 'ARCHAEA': 2, 'NEGATIVE': 3},
                     {'NO_SP': 0, 'SP': 1, 'TATLIPO': 2, 'LIPO': 3, 'TAT': 4, 'PILIN': 5},
                     {'V': 0, 'R': 1, 'D': 2, 'E': 3, 'H': 4, 'A': 5, 'G': 6, 'Y': 7, 'W': 8, 'F': 9, 'M': 10, 'K': 11,
                      'L': 12,
                      'I': 13, 'C': 14, 'Q': 15, 'S': 16, 'P': 17, 'N': 18, 'T': 19, 'PD': 20, 'BS': 21, 'ES': 22}]
        if form_sp_reg_data:
            pickle.dump(dicts, open("sp6_dicts_subregion_lbls.bin", "wb"))
        else:
            pickle.dump(dicts, open("sp6_dicts.bin", "wb"))

    def get_subregions_labels(self, seq, lbls, glbl_lbl="SP"):
        kyte_doolittle_hydrophobicity = {"A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8, "G": -0.4, "H": -3.2,
                                         "I": 4.5, "K": -3.9,
                                         "L": 3.8, "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5, "S": -0.8,
                                         "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3}

        def get_pos_rr_motif_v2(s, l):
            if "SRR" in s[:l.rfind("T") - 3] or "TRR" in s[:l.rfind("T") - 3]:
                return re.findall("[ST]RR", s[:l.rfind("T")])[0]
            elif len(re.findall("RR.F", s[:l.rfind("T") - 3])) != 0:
                return re.findall("RR.F", s)[0]
            elif len(re.findall("[R][RNKQ][DERKHNQSTYGAV]", s[:l.rfind("T") - 3])) != 0:
                # print(re.findall("[R][RNKQ][DERKHNQSTYGAV]", s[:l.rfind("T")]))
                return re.findall("[R][RNKQ][DERKHNQSTYGAV]", s[:l.rfind("T") - 3])[0]
            elif len(re.findall("[RK][R][DERKHNQSTYGAV]", s)) != 0:
                return re.findall("[RK][R][DERKHNQSTYGAV]", s[:l.rfind("T") - 3])[0]
            elif "RR" in s:
                return "RR"
            elif "MNDAAPQNPGQDEAKGTGEKDNGGSMSPRSALRTTAGVAGAGLGLSALGTGTASASVPEAAQTAVPAAES" == s:
                return "RS"

        def get_pos_of_rr_motif(seq_, lbls_, l_g=None):
            seq_ = seq_[:lbls_.rfind("T")]
            mtfs_found = []

            # for each motif,
            # 2 conditions: first, try to match the RRxF
            # second, try to see if K is after 4 aas

            def filter_F(current_seq, motifs):
                filtered_motifs = []
                for m in motifs:
                    if current_seq.find(m) + 6 + 3 < len(current_seq):
                        if current_seq[current_seq.find(m) + 2 + 4] == "F":
                            filtered_motifs.append(m)
                if len(filtered_motifs) == 0:
                    return motifs, True
                elif len(filtered_motifs) > 1:
                    return filtered_motifs, True
                elif len(filtered_motifs) == 1:
                    return filtered_motifs[0], False

            def filter_st(current_seq, motifs):
                filtered_motifs = []
                for previous in ["S", "T"]:
                    for m in motifs:
                        if current_seq[current_seq.find(m) - 1] == previous:
                            filtered_motifs.append(m)
                if len(filtered_motifs) == 0:
                    return motifs, True
                elif len(filtered_motifs) > 1:
                    return filtered_motifs, True
                elif len(filtered_motifs) == 1:
                    return filtered_motifs[0], False

            def get_first_RR_occ(current_seq, mtfs_):
                for m in mtfs_:
                    if "RR" in mtfs_ and m in current_seq:
                        return m
                return mtfs_[0]

            still_needs_filtering = True
            if "RR" in seq_ and len([m.start() for m in re.finditer("(?=RR)", seq_)]) == 1:
                mtfs_found = "RR"
                still_needs_filtering = False
            else:
                for first_RR in ["R", "K"]:
                    for second_RR in ["R", "N", "K", "Q"]:
                        for following_aa in ["D", "E", "R", "K", "H", "N", "Q", "S", "T", "Y", "G", "A",
                                             "V"]:
                            motif = first_RR + second_RR + following_aa
                            if motif in seq_ and seq_.find(motif) < len(seq_) - 6 and seq_.find(motif) > 1:
                                mtfs_found.append(motif)
                if len(mtfs_found) == 1:
                    mtfs_found = mtfs_found[0]
                    still_needs_filtering = False
            if len(mtfs_found) == 0:
                mtfs_found = seq[seq_.find("R"):seq_.find("R") + 2]
                still_needs_filtering = False
            if still_needs_filtering:
                mtfs_found, still_needs_filtering = filter_st(seq_, mtfs_found)
            if still_needs_filtering:
                mtfs_found, still_needs_filtering = filter_F(seq_, mtfs_found)
            if still_needs_filtering:
                mtfs_found = get_first_RR_occ(seq_, mtfs_found)
            if type(mtfs_found) == list:
                return mtfs_found[0]
            return mtfs_found

            # R/K; R/N/K/Q
            # usually RR
            # first can also be K
            # second can be also be N, K, Q
            # after motif: D,E, R, K, H , N, Q, S, T, Y, G, F

        def get_hydro_values(seq_, lbls_, sp_aa_lbl="S", start_ind=3):

            last_ind = lbls_.rfind(sp_aa_lbl)
            hydro_vals = []
            for i in range(start_ind, last_ind - 2):
                # window of 7 is used for the hydro values, determining the h region
                # compute these for all SP labels, except last 3 which are
                hydro_vals.append(sum([kyte_doolittle_hydrophobicity[seq_[j]] for j in range(i - 3, i + 4)]))
            h_ind = np.argmax(hydro_vals) + start_ind
            return h_ind, last_ind

        possible_sp_letters = ["S", "T", "L", "P"]
        #    sptype2letter = {'TAT':'T', 'LIPO':'L', 'PILIN':'P', 'TATLIPO':'T', 'SP':'S'}
        sp_letter = lbls[0]
        modified_lbls = ""
        # print(seq)
        # print(lbls)
        # print("\n")
        if sp_letter in possible_sp_letters:
            if glbl_lbl == "SP":
                h_ind, last_ind = get_hydro_values(seq, lbls, sp_aa_lbl=sp_letter)
                modified_lbls += "S" * 2 if self.simplified else "n" * 2
                modified_lbls += "S" * (h_ind - 2) if self.simplified else "N" * (h_ind - 2)
                modified_lbls += "S" if self.simplified else "h"
                modified_lbls += "S" * (last_ind - h_ind - 3) if self.simplified else "H" * (last_ind - h_ind - 3)
                modified_lbls += "S" * 3 if self.simplified else "c" * 3
                modified_lbls += lbls[last_ind + 1:]
            elif glbl_lbl == "LIPO":
                h_ind, last_ind = get_hydro_values(seq, lbls, sp_aa_lbl=sp_letter)
                modified_lbls += "S" * 2 if self.simplified else "n" * 2
                modified_lbls += "S" * (h_ind - 2) if self.simplified else "N" * (h_ind - 2)
                modified_lbls += "S" if self.simplified else "h"
                modified_lbls += "S" * (last_ind - h_ind - 3) if self.simplified else "h" * (last_ind - h_ind - 3)
                modified_lbls += "S" * 3 if self.simplified else "B" * 3
                # in LIPO/TATLIPO, there is always a cysteine aa after CS
                modified_lbls += "C" if not self.very_simplified else lbls[last_ind + 1]
                modified_lbls += lbls[last_ind + 2:]
            elif glbl_lbl == "TAT":
                # motif = get_pos_of_rr_motif(seq, lbls)
                motif = get_pos_rr_motif_v2(seq, lbls)
                # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2654714/
                # RR motif is a little more complicated
                h_ind, last_ind = get_hydro_values(seq, lbls, sp_aa_lbl=sp_letter, start_ind=seq.find(motif) + 3)
                # nnnnnnnRRn (n region has the RR motif and one single n after).
                modified_lbls = "S" * seq.find(motif) if self.simplified else "n" * seq.find(motif)
                modified_lbls += "R" * 2 if not self.very_simplified else "S" * 2
                modified_lbls += "S" if self.simplified else "n"
                current_len = len(modified_lbls)
                # add nh <- N (uncertain n or h label) until the most hydrophobic aa at h_ind
                modified_lbls += "S" * (h_ind - current_len) if self.simplified else "h" * (h_ind - current_len)
                # certain h subregion label at the most hydrophobic aa
                modified_lbls += "S" if self.simplified else "h"
                # uncertain hc <- H label until the last 3 aa in the signal peptide
                modified_lbls += "S" * (last_ind - h_ind - 3) if self.simplified else "H" * (last_ind - h_ind - 3)
                # certain c label subregion for the last 3 aa
                modified_lbls += "S" * 3 if self.simplified else "c" * 3
                modified_lbls += lbls[last_ind + 1:]
            elif glbl_lbl == "TATLIPO":
                # motif = get_pos_of_rr_motif(seq, lbls)
                motif = get_pos_rr_motif_v2(seq, lbls)
                h_ind, last_ind = get_hydro_values(seq, lbls, sp_aa_lbl=sp_letter, start_ind=seq.find(motif) + 3)
                # nnnnnnnRRn (n region has the RR motif and one single n after).
                modified_lbls = "S" * seq.find(motif) if self.simplified else "n" * seq.find(motif)
                modified_lbls += "R" * 2 if not self.very_simplified else "S" * 2
                modified_lbls += "S" if self.simplified else "n"
                current_len = len(modified_lbls)
                # add nh <- N (uncertain n or h label) until the most hydrophobic aa at h_ind
                modified_lbls += "S" * (h_ind - current_len) if self.simplified else "h" * (h_ind - current_len)
                # certain h subregion label at the most hydrophobic aa
                modified_lbls += "S" if self.simplified else "h"
                # uncertain hc <- H label until the last 3 aa in the signal peptide
                modified_lbls += "S" * (last_ind - h_ind - 3) if self.simplified else "h" * (last_ind - h_ind - 3)
                # certain c label subregion for the last 3 aa
                modified_lbls += "S" * 3 if self.simplified else "B" * 3
                modified_lbls += "C" if not self.very_simplified else lbls[last_ind + 1]
                modified_lbls += lbls[last_ind + 2:]
            elif glbl_lbl == "PILIN":
                return lbls.replace("P", "S")

            return modified_lbls
        else:
            return lbls

    def retrieve_raw_lbls(self):
        raw_data_file = self.get_data_folder() + "train_set.fasta"
        seq2lbls = {}
        for seq_record in SeqIO.parse(raw_data_file, "fasta"):
            current_seq = seq_record.seq[:len(seq_record.seq) // 2]
            if current_seq not in seq2lbls:
                seq2lbls[current_seq] = seq_record.seq[len(seq_record.seq) // 2:]
        return seq2lbls

    def form_subregion_sp_data(self):
        """ This function modifies the existing binaries containign the embeddings, with aa level labels corresponding
         to sp-subregions (n ,h, c) and others"""
        for tr_f in [0, 1, 2]:
            for t_set in ["train", "test"]:
                sp_subregion_data = {}
                data = pickle.load(
                    open(self.get_data_folder() + "sp6_partitioned_data_{}_{}.bin".format(t_set, tr_f), "rb"))
                for seq, (emb, lbls, l_grp, sp_type) in data.items():
                    sp_subregion_data[seq] = [emb, self.get_subregions_labels(seq, lbls, glbl_lbl=sp_type), l_grp,
                                              sp_type]
                pickle.dump(sp_subregion_data,
                            open(self.get_data_folder() + "sp6_partitioned_data_sublbls_{}_{}.bin".format(t_set, tr_f),
                                 "wb"))

    def form_lbl_inds(self):
        parts = [0, 1, 2]
        all_unique_lbls = set()
        unique_aas = set()
        for p in parts:
            for t in ["train", "test"]:
                part_dict = pickle.load(open(self.data_folder + "sp6_partitioned_data_{}_{}.bin".format(t, p), "rb"))
                for seq, (_, lbls, _, _) in part_dict.items():
                    all_unique_lbls.update(lbls)
                    unique_aas.update(seq)
        self.aa2ind = {aa: ind for ind, aa in enumerate(unique_aas)}
        self.lbl2ind = {l: ind for ind, l in enumerate(all_unique_lbls)}
        unique_aas = len(self.aa2ind.keys())
        PAD_IDX, BOS_IDX, EOS_IDX = unique_aas, unique_aas + 1, unique_aas + 2
        self.aa2ind['PD'] = PAD_IDX
        self.aa2ind["BS"] = BOS_IDX
        self.aa2ind["ES"] = EOS_IDX
        # add special tokens at the end of the dictionary
        unique_tkns = len(self.lbl2ind.keys())
        PAD_IDX, BOS_IDX, EOS_IDX = unique_tkns, unique_tkns + 1, unique_tkns + 2
        self.lbl2ind["PD"] = PAD_IDX
        self.lbl2ind["BS"] = BOS_IDX
        self.lbl2ind["ES"] = EOS_IDX

        all_unique_lgs = set()
        all_unique_global_inds = set()
        for p in parts:
            for t in ["train", "test"]:
                part_dict = pickle.load(open(self.data_folder + "sp6_partitioned_data_{}_{}.bin".format(t, p), "rb"))
                for (_, _, lg, glb_ind) in part_dict.values():
                    all_unique_lgs.add(lg)
                    all_unique_global_inds.add(glb_ind)
        self.lg2ind = {l: ind for ind, l in enumerate(all_unique_lgs)}
        self.glbl_lbl_2ind = {l: ind for ind, l in enumerate(all_unique_global_inds)}
        pickle.dump([self.lbl2ind, self.lg2ind, self.glbl_lbl_2ind, self.aa2ind], open("sp6_dicts.bin", "wb"))

    def get_data_folder(self):
        if os.path.exists("/scratch/work/dumitra1"):
            return "/scratch/work/dumitra1/sp_data/"
        elif os.path.exists("/home/alex"):
            return "sp_data/"
        else:
            return "/scratch/project2003818/dumitra1/sp_data/"


class CSPredsDataset(Dataset):
    def __init__(self, lbl2inds, partitions, data_folder, glbl_lbl_2ind, train=True, sets=["train", "test"],
                 test_f_name="",
                 form_sp_reg_data=False):
        self.life_grp, self.seqs, self.lbls, self.glbl_lbl = [], [], [], []
        if partitions is not None:
            # when using partitions, the sp6 data partition files will be used in train/testing
            for p in partitions:
                for s in sets:
                    d_file = "sp6_partitioned_data_sublbls_{}_{}.bin".format(s, p) if form_sp_reg_data else \
                        "sp6_partitioned_data_{}_{}.bin".format(s, p)

                    data_dict = pickle.load(open(data_folder + d_file, "rb"))
                    self.seqs.extend(list(data_dict.keys()))
                    self.lbls.extend([[lbl2inds[l] for l in label] for (_, label, _, _) in data_dict.values()])
                    self.life_grp.extend([life_grp for (_, _, life_grp, _) in data_dict.values()])
                    self.glbl_lbl.extend([glbl_lbl_2ind[glbl_lbl] for (_, _, _, glbl_lbl) in data_dict.values()])
        else:
            # parameter for a specific test
            data_dict = pickle.load(open(data_folder + test_f_name, "rb"))
            self.seqs.extend(list(data_dict.keys()))
            self.lbls.extend([[lbl2inds[l] for l in label] for (_, label, _, _) in data_dict.values()])
            self.life_grp.extend([life_grp for (_, _, life_grp, _) in data_dict.values()])
            self.glbl_lbl.extend([glbl_lbl_2ind[glbl_lbl] for (_, _, _, glbl_lbl) in data_dict.values()])

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, item):
        return {"seq": self.seqs[item], "lbl": self.lbls[item], "lg": self.life_grp[item],
                "glbl_lbl": self.glbl_lbl[item]}


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
    src_batch, tgt_batch, life_grp, glbl_lbl = [], [], [], []
    for sample in batch:
        src_batch.append(sample['seq'])
        tgt_batch.append(sample['lbl'])
        life_grp.append(sample['lg'])
        glbl_lbl.append(sample['glbl_lbl'])
    return src_batch, tgt_batch, life_grp, glbl_lbl


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


def get_data_folder():
    if os.path.exists("/scratch/work/dumitra1"):
        return "/scratch/work/dumitra1/sp_data/"
    elif os.path.exists("/home/alex"):
        return "sp_data/"
    else:
        return "/scratch/project2003818/dumitra1/sp_data/"


def get_sp_type_loss_weights():
    data_folder = get_data_folder()
    sptye2count = {'NO_SP': 0, 'SP': 0, 'TATLIPO': 0, 'LIPO': 0, 'TAT': 0, 'PILIN': 0}

    for tr_f in [0, 1, 2]:
        for set in ["train", "test"]:
            data = pickle.load(open(data_folder+"sp6_partitioned_data_sublbls_{}_{}.bin".format(set, tr_f), "rb"))
            for sp_t in data.values():
                sptye2count[sp_t[3]] += 1
    min_count = min(sptye2count.values())
    return {k:min_count/v for k,v in sptye2count.items()}

def get_residue_label_loss_weights():
    data_folder = get_data_folder()
    aalbl2count = {'S': 0, 'O': 0, 'M': 0, 'I': 0} #,

    for tr_f in [0, 1, 2]:
        for set in ["train", "test"]:
            data = pickle.load(open(data_folder + "sp6_partitioned_data_sublbls_{}_{}.bin".format(set, tr_f), "rb"))
            for seq in data.values():
                seq_ = seq[1]
                for r in seq_:
                    aalbl2count[r] += 1
    min_count = min(aalbl2count.values())
    #'PD': 4, 'BS': 5, 'ES': 6}
    lbl2weights = {k:min_count/v for k,v in aalbl2count.items()}
    lbl2weights.update({'PD':1, 'BS':1, 'ES':1})
    return lbl2weights
