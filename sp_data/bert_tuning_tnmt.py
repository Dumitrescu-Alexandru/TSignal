# 4. function that gives a loaded model
# 4. function that takes in the model and some sequences and gives embeddings

# 3. give file and columns (for cdr3/TCR seqs, and delimiter)
from torch.nn import LayerNorm
import logging
import datetime
# from transformers import AutoTokenizer, AutoModel, pipeline
import pickle
import random
import sys
import os
sys.path.append(os.path.abspath(".."))
from misc.visualize_cs_pred_results import get_cs_and_sp_pred_results, get_summary_sp_acc, get_summary_cs_acc
from train_scripts.cv_train_cs_predictors import log_and_print_mcc_and_cs_results, modify_sp_subregion_preds, clean_sec_sp2_preds, padd_add_eos_tkn
from models.transformer_nmt import TransformerModel
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, RandomSampler
from models.transformer_nmt import TokenEmbedding, PositionalEncoding
from torch.nn import TransformerDecoder, TransformerDecoderLayer
import pytorch_lightning as pl
from pytorch_lightning.loggers import TestTubeLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning import Trainer, seed_everything

from transformers import BertTokenizer, BertModel

from torchnlp.encoders import LabelEncoder
from torchnlp.datasets.dataset import Dataset
from torchnlp.utils import collate_tensors
import pandas as pd
from test_tube import HyperOptArgumentParser
import os
import requests
from tqdm.auto import tqdm
from datetime import datetime
from collections import OrderedDict
import logging as log
import numpy as np

def gpu_acc_metric(y_hat, labels):
    # A torch way of extracting accuracy. Used like this for gpu compatibility
    with torch.no_grad():
        acc = sum(y_hat == torch.tensor(labels, device=y_hat.device)).to(dtype=torch.float) / y_hat.shape[0]
    return acc


def extract_seq_lbls(folds=[0, 1], t_set="train", relative_data_path="", use_glbl_labels=False):
    prefix = "sublbls_" if use_glbl_labels else ""
    data = pickle.load(open(relative_data_path + "sp6_partitioned_data_" + prefix + "{}_{}.bin".format(t_set, folds[0]), "rb"))
    seqs = list(data.keys())
    lbls = [data[s][1] for s in seqs]
    glbl_lbls = [data[s][3] for s in seqs]
    sp_types = [data[s][-1] for s in seqs]
    data = pickle.load(open(relative_data_path + "sp6_partitioned_data_" + prefix + "{}_{}.bin".format(t_set, folds[1]), "rb"))
    keys_2nd_data = list(data.keys())
    seqs.extend(keys_2nd_data)
    lbls.extend(data[s][1] for s in keys_2nd_data)
    glbl_lbls.extend(data[s][3] for s in keys_2nd_data)
    sp_types.extend([data[s][-1] for s in keys_2nd_data])
    if len(folds) == 3:
        data = pickle.load(open(relative_data_path + "sp6_partitioned_data_" + prefix + "{}_{}.bin".format(t_set, folds[2]), "rb"))
        keys_3rd_data = list(data.keys())
        seqs.extend(keys_3rd_data)
        lbls.extend(data[s][1] for s in keys_3rd_data)
        sp_types.extend([data[s][-1] for s in keys_3rd_data])
        glbl_lbls.extend(data[s][3] for s in keys_2nd_data)

    return seqs, lbls, glbl_lbls

def create_sp6_training_ds(relative_data_path, folds=[0, 1], use_glbl_labels=False):
    agnostic_lbls_identifier = "sublbls_" if use_glbl_labels else ""
    train_seqs, train_lbls, train_glbl_lbls = extract_seq_lbls(folds, "train", relative_data_path, use_glbl_labels=use_glbl_labels)
    test_seqs, test_lbls, test_glbl_lbls = extract_seq_lbls(folds, "test", relative_data_path, use_glbl_labels=use_glbl_labels)

    if use_glbl_labels:
        train_df = pd.DataFrame({'seqs': train_seqs, 'lbls': train_lbls, 'glbl_lbls':train_glbl_lbls})
        test_df = pd.DataFrame({'seqs': test_seqs, 'lbls': test_lbls, 'glbl_lbls':test_glbl_lbls})
        valid_df = pd.DataFrame({'seqs': test_seqs, 'lbls': test_lbls, 'glbl_lbls':test_glbl_lbls})
    else:
        train_df = pd.DataFrame({'seqs': train_seqs, 'lbls': train_lbls})
        test_df = pd.DataFrame({'seqs': test_seqs, 'lbls': test_lbls})
        valid_df = pd.DataFrame({'seqs': test_seqs, 'lbls': test_lbls})

    if len(folds) == 3:
        train_df.to_csv(relative_data_path + "sp6_fine_tuning_train_"+ agnostic_lbls_identifier +"{}_{}_{}.csv".format(*folds))
        test_df.to_csv(relative_data_path + "sp6_fine_tuning_test_"+ agnostic_lbls_identifier +"{}_{}_{}.csv".format(*folds))
        valid_df.to_csv(relative_data_path + "sp6_fine_tuning_valid_"+ agnostic_lbls_identifier+ "{}_{}_{}.csv".format(*folds))
    else:
        train_df.to_csv(relative_data_path + "sp6_fine_tuning_train_" + agnostic_lbls_identifier +"{}_{}.csv".format(*folds))
        test_df.to_csv(relative_data_path + "sp6_fine_tuning_test_"+ agnostic_lbls_identifier +"{}_{}.csv".format(*folds))
        valid_df.to_csv(relative_data_path + "sp6_fine_tuning_valid_"+ agnostic_lbls_identifier +"{}_{}.csv".format(*folds))


def create_sp6_tuning_dataset(relative_data_path, folds=[0, 1]):
    data = pickle.load(open(relative_data_path + "sp6_partitioned_data_train_{}.bin".format(folds[0]), "rb"))
    seqs = list(data.keys())
    lbls = [data[s][1] for s in seqs]
    sp_types = [data[s][-1] for s in seqs]
    data = pickle.load(open(relative_data_path + "sp6_partitioned_data_train_{}.bin".format(folds[1]), "rb"))
    keys_2nd_data = list(data.keys())
    seqs.extend(keys_2nd_data)
    lbls.extend(data[s][1] for s in keys_2nd_data)
    sp_types.extend([data[s][-1] for s in keys_2nd_data])
    if len(folds) == 3:
        data = pickle.load(open(relative_data_path + "sp6_partitioned_data_train_{}.bin".format(folds[2]), "rb"))
        keys_3rd_data = list(data.keys())
        seqs.extend(keys_3rd_data)
        lbls.extend(data[s][1] for s in keys_3rd_data)
        sp_types.extend([data[s][-1] for s in keys_3rd_data])

    sp_types_2inds = {s: [] for s in set(sp_types)}
    train_seqs, train_lbls, test_seqs, test_lbls = [], [], [], []
    for ind, s_t in enumerate(sp_types):
        sp_types_2inds[s_t].append(ind)
    for sp_t, inds in sp_types_2inds.items():
        test_inds_ = random.sample(inds, max(2, round(len(inds) * 0.1)))
        test_seqs.extend([seqs[i] for i in test_inds_])
        test_lbls.extend([lbls[i] for i in test_inds_])
        train_seqs.extend([seqs[i] for i in set(inds) - set(test_inds_)])
        train_lbls.extend([lbls[i] for i in set(inds) - set(test_inds_)])
    train_df = pd.DataFrame({'seqs': train_seqs, 'lbls': train_lbls})
    test_df = pd.DataFrame({'seqs': test_seqs, 'lbls': test_lbls})
    valid_df = pd.DataFrame({'seqs': test_seqs, 'lbls': test_lbls})
    if len(folds) == 3:
        train_df.to_csv(relative_data_path + "sp6_fine_tuning_train_{}_{}_{}.csv".format(*folds))
        test_df.to_csv(relative_data_path + "sp6_fine_tuning_test_{}_{}_{}.csv".format(*folds))
        valid_df.to_csv(relative_data_path + "sp6_fine_tuning_valid_{}_{}_{}.csv".format(*folds))
    else:
        train_df.to_csv(relative_data_path + "sp6_fine_tuning_train_{}_{}.csv".format(*folds))
        test_df.to_csv(relative_data_path + "sp6_fine_tuning_test_{}_{}.csv".format(*folds))
        valid_df.to_csv(relative_data_path + "sp6_fine_tuning_valid_{}_{}.csv".format(*folds))


def create_epitope_tuning_files(relative_data_path):
    """
        Creates input sequences and epitope specificity label dataset for task fine-tuning and saves the files in the
        data folder.
            Inputs:
                relative_data_path: path from the main script to the data folder
    """

    def split_data(eps):
        ep2id = {}
        for ind, e in enumerate(eps):
            if e not in ep2id:
                ep2id[e] = [ind]
            else:
                ep2id[e].append(ind)
        all_test_inds, all_val_inds, all_train_inds = [], [], []
        for _, inds in ep2id.items():
            test_inds_ = random.sample(inds, max(2, round(len(inds) * 0.05)))
            val_inds_ = test_inds_
            train_inds_ = [i for i in inds if i not in test_inds_]
            all_test_inds.extend(test_inds_)
            all_train_inds.extend(train_inds_)
            all_val_inds.extend(val_inds_)
        return all_train_inds, all_val_inds, all_test_inds

    epitope_dataset = pd.read_csv(relative_data_path + 'vdj_human_unique_longs.csv')
    epitope, sequences = epitope_dataset['epitope'].values, epitope_dataset['long'].values
    all_train_inds, all_val_inds, all_test_inds = split_data(epitope)
    train_seq, train_ep, test_seq, test_ep = sequences[all_train_inds], epitope[all_train_inds], \
                                             sequences[all_test_inds], epitope[all_test_inds]

    train_df = pd.DataFrame(train_seq, train_ep)
    test_df = pd.DataFrame(test_seq, test_ep)
    valid_df = pd.DataFrame(test_seq, test_ep)

    train_df.to_csv(relative_data_path + "epitope_seq_train.csv")
    test_df.to_csv(relative_data_path + "epitope_seq_test.csv")
    valid_df.to_csv(relative_data_path + "epitope_seq_valid.csv")


def create_bert_further_tuning_files(relative_data_path="."):
    """
        Extracts raw sequences for further tuning the BERT model on TCR-only data. The 15% masking is done
        in the dataset class, later
    """
    data = pd.read_csv(os.path.join(relative_data_path, "vdj_human_unique_longs.csv"))
    seqs = data['long'].values
    train_format_seqs = []
    for s in seqs:
        train_format_seqs.append(" ".join([s_ for s_ in s]))
    test_inds = random.sample(list(range(len(train_format_seqs))), int(0.2 * len(train_format_seqs)))
    valid_seqs, test_seqs = test_inds[:len(test_inds) // 2], test_inds[len(test_inds) // 2:]
    train_seqs = list(set(list(range(len(train_format_seqs)))) - set(test_inds))
    valid_data, test_data, train_data = [train_format_seqs[i] for i in valid_seqs], \
                                        [train_format_seqs[i] for i in test_seqs], \
                                        [train_format_seqs[i] for i in train_seqs]
    valid_data, test_data, train_data = pd.DataFrame(valid_data), pd.DataFrame(test_data), pd.DataFrame(train_data)
    train_data.to_csv(os.path.join(relative_data_path, "tcr_seqs_train_df.csv"))
    valid_data.to_csv(os.path.join(relative_data_path, "tcr_seqs_dev_df.csv"))
    test_data.to_csv(os.path.join(relative_data_path, "tcr_seqs_test_df.csv"))


class EpitopeClassifierDataset(Dataset):
    """
        Dataset class for BERT tuning with the epitope specificity classification. A <CLS> token added at the beginning
        of the sequences will be used as a representation of the whole sequence, and the classification will be based
        on the hidden representation of that token.
    """

    def __init__(self, raw_path, file) -> None:
        self.data = []
        vocab = pickle.load(open(raw_path + "lbl2vocab.bin", "rb"))
        path = raw_path + file
        self.init_dataset(path, vocab)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        seq = self.data.iloc[item, 0]
        target = self.data.iloc[item, 1]
        sample = {"seq": seq, "target": target}
        return sample

    def collate_lists(self, seq: list, label: list) -> dict:
        """ Converts each line into a dictionary. """
        collated_dataset = []
        for i in range(len(seq)):
            collated_dataset.append({"seq": str(seq[i]), "label": label[i]})
        return collated_dataset

    def calculate_stat(self, path):
        df = pd.read_csv(path, names=['input', 'loc', 'membrane'], skiprows=1)
        df = df.loc[df['membrane'].isin(["M", "S"])]

        self.nSamples_dic = df['membrane'].value_counts()

    def load_dataset(self, path):
        self.init_dataset(path)
        return self

    def init_dataset(self, path, vocab):
        df = pd.read_csv(path)
        label, sequences = [], []
        for ep, seq in df.values:
            label.append(vocab[ep])
            sequences.append(" ".join([s for s in seq]))
        assert len(sequences) == len(label)
        self.data = pd.DataFrame({"seq": sequences, "label": label})


class SP6TuningDataset(Dataset):
    """
        Dataset class for BERT tuning with the epitope specificity classification. A <CLS> token added at the beginning
        of the sequences will be used as a representation of the whole sequence, and the classification will be based
        on the hidden representation of that token.
    """

    def __init__(self, raw_path, file, use_glbl_lbls=False) -> None:
        self.data = []
        vocab = {'P': 0, 'S': 1, 'O': 2, 'M': 3, 'L': 4, 'I': 5, 'T': 6, 'W': 7, 'PD': 8, 'BS': 9,
                                 'ES': 10} if not use_glbl_lbls else \
            {'S': 0, 'O': 1, 'M': 2, 'I': 3, 'PD': 4, 'BS': 5, 'ES': 6}
        self.glbl_vocab = {'NO_SP': 0, 'SP': 1, 'TATLIPO': 2, 'LIPO': 3, 'TAT': 4, 'PILIN': 5}
        path = raw_path + file
        self.use_glbl_lbls = use_glbl_lbls
        self.init_dataset(path, vocab)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        seq = self.data.iloc[item, 0]
        target = self.data.iloc[item, 1]
        if self.use_glbl_lbls:
            glbl_target = self.data.iloc[item, 2]
            sample = {"seq": seq, "target": (target, glbl_target)}
        else:
            sample = {"seq": seq, "target": target}
        return sample

    def collate_lists(self, seq: list, label: list) -> dict:
        """ Converts each line into a dictionary. """
        collated_dataset = []
        for i in range(len(seq)):
            collated_dataset.append({"seq": str(seq[i]), "label": label[i]})
        return collated_dataset

    def calculate_stat(self, path):
        df = pd.read_csv(path, names=['input', 'loc', 'membrane'], skiprows=1)
        df = df.loc[df['membrane'].isin(["M", "S"])]

        self.nSamples_dic = df['membrane'].value_counts()

    def load_dataset(self, path):
        self.init_dataset(path)
        return self

    def init_dataset(self, path, vocab):
        if self.use_glbl_lbls:
            df = pd.read_csv(path)
            label, sequences, glbl_lbls = [], [], []
            for seq, seq_lbl, glbl_lbl in zip(df['seqs'].values, df['lbls'].values, df['glbl_lbls'].values):
                label.append([vocab[l] for l in seq_lbl])
                sequences.append(" ".join([s for s in seq]))
                glbl_lbls.append(self.glbl_vocab[glbl_lbl])
            assert len(sequences) == len(label)
            self.data = pd.DataFrame({"seq": sequences, "label": label, "glbl_lbls": glbl_lbls})
        else:
            df = pd.read_csv(path)
            label, sequences = [], []
            for seq, seq_lbl in zip(df['seqs'].values, df['lbls'].values):
                label.append([vocab[l] for l in seq_lbl])
                sequences.append(" ".join([s for s in seq]))
            assert len(sequences) == len(label)
            self.data = pd.DataFrame({"seq": sequences, "label": label})

class BertDataset(Dataset):
    """
    Loads the Dataset from the csv files passed to the parser. Extracts 15% of residues in seequences and replaces them
    with <MASK>. Also retains the masked indices - the respective indices will be used later to extract the masked
    residues and predict the missing residues.
            Inputs:
                file: csv file containing only raw sequences
                special_tokens: whether the model uses special tokens like <CLS>: (shift the mask indices in that case)
                relative_data_path: path to file
    """

    def __init__(self, file, special_tokens, relative_data_path) -> None:
        self.data = []
        self.init_dataset(os.path.join(relative_data_path, file), special_tokens)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        seqs = self.data.iloc[item, 0]
        target = self.data.iloc[item, 1]
        target_pos = self.data.iloc[item, 2]
        sample = {"seq": seqs, "target": target, "target_pos": target_pos}
        return sample

    def collate_lists(self, seq: list, label: list, label_inds: list) -> dict:
        """ Converts each line into a dictionary. """
        collated_dataset = []
        for i in range(len(seq)):
            collated_dataset.append({"seq": str(seq[i]), "label": label[i], "label_inds": label_inds[i]})
        return collated_dataset

    def calculate_stat(self, path):
        df = pd.read_csv(path, names=['input', 'loc', 'membrane'], skiprows=1)
        df = df.loc[df['membrane'].isin(["M", "S"])]

        self.nSamples_dic = df['membrane'].value_counts()

    def load_dataset(self, path):
        self.init_dataset(path)
        return self

    def init_dataset(self, path, special_tokens):
        def create_labels(sequences):
            all_lbls, all_pos, masked_s = [], [], []
            for s in sequences:
                if type(s) == str:
                    current_s = s.split(" ")
                    cs = s.replace(" ", "")
                    no_masks = int((15 / 100) * len(cs))
                    inds = random.sample(list(range(len(cs))), no_masks)
                    lbl, lbl_pos = [], []
                    for i in inds:
                        lbl.append(cs[i])
                        if special_tokens:
                            lbl_pos.append(i + 1)
                        else:
                            lbl_pos.append(i)
                        current_s[i] = "[MASK]"
                    all_lbls.append(lbl)
                    all_pos.append(lbl_pos)
                    masked_s.append(" ".join(current_s))
            return all_lbls, all_pos, masked_s

        df = pd.read_csv(path, names=['sequences'], skiprows=1)

        seq = list(df['sequences'])
        # label = list(df['membrane'])
        label, label_inds, seq = create_labels(seq)
        assert len(seq) == len(label)
        assert len(seq) == len(label_inds)
        self.data = pd.DataFrame({"seq": seq, "label": label, "label_inds": label_inds})


class ProtBertClassifier(pl.LightningModule):
    """
    # https://github.com/minimalist-nlp/lightning-text-classification.git

    Sample model to show how to use BERT to classify sentences.

    :param hparams: ArgumentParser containing the hyperparameters.
    """

    def __init__(self, hparams) -> None:
        super(ProtBertClassifier, self).__init__()
        self.hparams = hparams
        self.batch_size = self.hparams.batch_size
        self.e = 0

        if self.hparams.tune_sp6_labels or hparams.train_enc_dec_sp6:
            self.lbl2ind_dict = {'P': 0, 'S': 1, 'O': 2, 'M': 3, 'L': 4, 'I': 5, 'T': 6, 'W': 7, 'PD': 8, 'BS': 9,
                                 'ES': 10} if not hparams.use_glbl_labels else {'S': 0, 'O': 1, 'M': 2, 'I': 3, 'PD': 4, 'BS': 5, 'ES': 6}
            self.glbl_lbl2ind = {'NO_SP': 0, 'SP': 1, 'TATLIPO': 2, 'LIPO': 3, 'TAT': 4, 'PILIN': 5}
        if os.path.exists('/scratch/work/dumitra1'):
            self.modelFolderPath = "../../../covid_tcr_protein_embeddings/models/ProtBert/"
        else:
            self.modelFolderPath = './models/ProtBert/'
        self.vocabFilePath = os.path.join(self.modelFolderPath, 'vocab.txt')

        self.extract_emb = False
        self.metric_acc = gpu_acc_metric

        # build model
        self.__download_model()

        # build model
        self.__build_model()

        # Loss criterion initialization.
        self.__build_loss()

        if self.hparams.nr_frozen_epochs > 0:
            self.freeze_encoder()
        else:
            self._frozen = False
        self.nr_frozen_epochs = self.hparams.nr_frozen_epochs
        self.aaind2lblvocab = {v:k for k, v in self.tokenizer.get_vocab().items()}

    def __download_model(self) -> None:
        modelUrl = 'https://www.dropbox.com/s/dm3m1o0tsv9terq/pytorch_model.bin?dl=1'
        configUrl = 'https://www.dropbox.com/s/d3yw7v4tvi5f4sk/bert_config.json?dl=1'
        vocabUrl = 'https://www.dropbox.com/s/jvrleji50ql5m5i/vocab.txt?dl=1'

        modelFolderPath = self.modelFolderPath
        modelFilePath = os.path.join(modelFolderPath, 'pytorch_model.bin')
        configFilePath = os.path.join(modelFolderPath, 'config.json')
        vocabFilePath = self.vocabFilePath

        if not os.path.exists(modelFolderPath):
            os.makedirs(modelFolderPath)

        def download_file(url, filename):
            response = requests.get(url, stream=True)
            with tqdm.wrapattr(open(filename, "wb"), "write", miniters=1,
                               total=int(response.headers.get('content-length', 0)),
                               desc=filename) as fout:
                for chunk in response.iter_content(chunk_size=4096):
                    fout.write(chunk)

        if not os.path.exists(modelFilePath):
            download_file(modelUrl, modelFilePath)

        if not os.path.exists(configFilePath):
            download_file(configUrl, configFilePath)

        if not os.path.exists(vocabFilePath):
            download_file(vocabUrl, vocabFilePath)

    def __build_model(self) -> None:
        """ Init BERT model + tokenizer + classification head."""
        self.ProtBertBFD = BertModel.from_pretrained(self.modelFolderPath,
                                                     gradient_checkpointing=self.hparams.gradient_checkpointing)
        self.train_BFD = False if self.hparams.nr_frozen_epochs > 0 else True
        self.encoder_features = 1024

        # Tokenizer
        self.tokenizer = BertTokenizer(self.vocabFilePath, do_lower_case=False)

        # Label Encoder
        self.label_encoder = LabelEncoder(
            self.hparams.label_set.split(","), reserved_labels=[]
        )
        self.label_encoder.unknown_index = None
        if hparams.train_enc_dec_sp6:
            self.classification_head = TransformerModel(ntoken=len(self.lbl2ind_dict.keys()),
                                    lbl2ind=self.lbl2ind_dict,
                                    lg2ind={'EUKARYA': 0, 'POSITIVE': 1, 'ARCHAEA': 2, 'NEGATIVE': 3}, dropout=0.1,
                                    use_glbl_lbls=self.hparams.use_glbl_labels, no_glbl_lbls=6, ff_dim=4096, nlayers=3, nhead=16, aa2ind=None,
                                    train_oh=False, glbl_lbl_version=3, form_sp_reg_data=self.hparams.use_glbl_labels, version2_agregation="max",
                                    input_drop=False, no_pos_enc=hparams.no_pos_enc, linear_pos_enc=False, scale_input=False,
                                                        tuned_bert_embs_prefix="",tuning_bert=True, d_model = 1024, d_hid=1024)
            self.label_encoder_t_dec = TokenEmbedding(len(self.lbl2ind_dict.keys()), 1024, lbl2ind=self.lbl2ind_dict)
            self.pos_encoder = PositionalEncoding(1024)
        elif self.hparams.tune_sp6_labels:
            self.classification_head = nn.Sequential(
                nn.Linear(self.encoder_features, len(self.lbl2ind_dict.keys()))
            )
        elif self.hparams.tune_epitope_specificity:
            self.classification_head = nn.Sequential(
                nn.Linear(self.encoder_features * 4, 22),
                nn.Tanh()
            )
        else:
            self.classification_head = nn.Sequential(
                nn.Linear(self.encoder_features, 25),
                nn.Tanh()
            )

    @staticmethod
    def get_epitope_weights(relative_data_path):
        vdj_long_data = pd.read_csv(os.path.join(relative_data_path, "vdj_human_unique_longs.csv"))
        epitope2ind = pickle.load(open(os.path.join(relative_data_path, "lbl2vocab.bin"), "rb"))
        epitope2count = {}
        for ep in vdj_long_data['epitope'].values:
            if ep in epitope2count:
                epitope2count[ep] += 1
            else:
                epitope2count[ep] = 1
        ind_ep_2weights = {}
        n_samples = len(vdj_long_data['epitope'].values)
        n_classes = len(epitope2ind.keys())
        for epitope, ind in epitope2ind.items():
            ind_ep_2weights[ind] = n_samples / (n_classes * epitope2count[epitope])
        ordered_weights = []
        for ind in range(n_classes):
            ordered_weights.append(ind_ep_2weights[ind])
        return ordered_weights

    def __build_loss(self):
        """ Initializes the loss function/s. """
        if self.hparams.tune_sp6_labels or hparams.train_enc_dec_sp6:
            self._loss = nn.CrossEntropyLoss(ignore_index=self.lbl2ind_dict['PD'])
            if self.hparams.use_glbl_labels:
                self._loss2 = nn.CrossEntropyLoss()
        elif self.hparams.tune_epitope_specificity:
            weights = self.get_epitope_weights(self.hparams.relative_data_path)
            self._loss = nn.CrossEntropyLoss(weight=torch.tensor(weights))
        else:
            self._loss = nn.CrossEntropyLoss()

    def unfreeze_encoder(self) -> None:
        """ un-freezes the encoder layer. """
        self.train_BFD= True
        if self._frozen:
            log.info(f"\n-- Encoder model fine-tuning")
            for name, param in self.ProtBertBFD.named_parameters():
                param.requires_grad = True
            self._frozen = False

    def freeze_encoder(self) -> None:
        """ freezes the encoder layer. """
        for param in self.ProtBertBFD.parameters():
            param.requires_grad = False
        self._frozen = True

    def unfreeze_classification_head(self):
        for param in self.classification_head.parameters():
            param.requires_grad = True

    def predict(self, sample: dict) -> dict:
        """ Predict function.
        :param sample: dictionary with the text we want to classify.
        Returns:
            Dictionary with the input text and the predicted label.
        """
        if self.training:
            self.eval()

        with torch.no_grad():
            model_input, _ = self.prepare_sample([sample], prepare_target=False)
            model_out = self.forward(**model_input)
            logits = model_out["logits"].numpy()
            predicted_labels = [
                self.label_encoder.index_to_token[prediction]
                for prediction in np.argmax(logits, axis=1)
            ]
            sample["predicted_label"] = predicted_labels[0]

        return sample

    # https://github.com/UKPLab/sentence-transformers/blob/eb39d0199508149b9d32c1677ee9953a84757ae4/sentence_transformers/models/Pooling.py
    def pool_strategy(self, features,
                      pool_cls=True, pool_max=True, pool_mean=True,
                      pool_mean_sqrt=True):
        token_embeddings = features['token_embeddings']
        cls_token = features['cls_token_embeddings']
        attention_mask = features['attention_mask']

        ## Pooling strategy
        output_vectors = []
        if pool_cls:
            output_vectors.append(cls_token)
        if pool_max:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
            max_over_time = torch.max(token_embeddings, 1)[0]
            output_vectors.append(max_over_time)
        if pool_mean or pool_mean_sqrt:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)

            # If tokens are weighted (by WordWeights layer), feature 'token_weights_sum' will be present
            if 'token_weights_sum' in features:
                sum_mask = features['token_weights_sum'].unsqueeze(-1).expand(sum_embeddings.size())
            else:
                sum_mask = input_mask_expanded.sum(1)

            sum_mask = torch.clamp(sum_mask, min=1e-9)

            if pool_mean:
                output_vectors.append(sum_embeddings / sum_mask)
            if pool_mean_sqrt:
                output_vectors.append(sum_embeddings / torch.sqrt(sum_mask))

        output_vector = torch.cat(output_vectors, 1)
        return output_vector

    def extract_embeddnings(self, sample):
        inputs = self.tokenizer.batch_encode_plus(sample["seq"],
                                                  add_special_tokens=True,
                                                  padding=True,
                                                  truncation=True,
                                                  max_length=self.hparams.max_length)
        return self.forward(inputs['input_ids'], inputs['token_type_ids'], inputs['attention_mask'],
                            return_embeddings=True).cpu().numpy()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones((sz, sz), device=self.device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, input_ids, token_type_ids, attention_mask, target_positions=None, return_embeddings=False,
                targets=None, seq_lengths=None, v=False):
        """ Usual pytorch forward function.
        :param tokens: text sequences [batch_size x src_seq_len]
        :param lengths: source lengths [batch_size]
        Returns:
            Dictionary with model outputs (e.g: logits)
        """
        input_ids = torch.tensor(input_ids, device=self.device)
        inp_seqs = []
        for inp in input_ids:
            inp_seqs.append("".join([self.aaind2lblvocab[i_] for i_ in inp.detach().cpu().numpy()]).replace("[PAD]", ""))
        batch_size, seq_dim = input_ids.shape[0], input_ids.shape[1]
        attention_mask = torch.tensor(attention_mask, device=self.device)
        word_embeddings = self.ProtBertBFD(input_ids,
                                           attention_mask)[0]

        if self.extract_emb:
            # used for extracting the actual embeddings after tuning
            return word_embeddings

        if self.hparams.tune_epitope_specificity:
            # we dont want any pooling (only extract the embeddings corresponding to the masked inputs)
            pooling = self.pool_strategy({"token_embeddings": word_embeddings,
                                          "cls_token_embeddings": word_embeddings[:, 0],
                                          "attention_mask": attention_mask,
                                          })
            return self.classification_head(pooling)
        elif self.hparams.tune_sp6_labels:
            return self.classification_head(word_embeddings)
        elif self.hparams.train_enc_dec_sp6:
            # if v:
                # "MALTDGGWCLPKRFGAAGADASDSRAFPAREPSTPPSPISSSSSSCSRGGERGPGGASNCGTPQLDTEAA"
                # print( "".join(self.aaind2lblvocab[i_.item()] for i_ in input_ids[0]))
                # print(inp_seqs)
            return self.classification_head(word_embeddings, targets, inp_seqs=inp_seqs)



        word_embeddings = word_embeddings.reshape(-1, 1024)
        seq_delim = torch.tensor(list(range(batch_size)), device=self.device) * seq_dim
        seq_delim = seq_delim.reshape(-1, 1)
        target_positions = torch.tensor(target_positions, device=self.device).reshape(-1, len(target_positions[0]))
        target_positions = target_positions + seq_delim
        target_positions = target_positions.reshape(-1)
        prediction_embeddings = word_embeddings[target_positions]
        # return {"logits": self.classification_head(prediction_embeddings)}
        out = self.classification_head(prediction_embeddings)

        # return self.classification_head(prediction_embeddings)
        return out

    def loss(self, predictions: torch.tensor, targets: dict) -> torch.tensor:
        """
        Computes Loss value according to a loss function.
        :param predictions: model specific output. Must contain a key 'logits' with
            a tensor [batch_size x 1] with model predictions
        :param labels: Label values [batch_size]
        Returns:
            torch.tensor with loss value.
        """
        # return self._loss(predictions["logits"], torch.tensor(targets["labels"], device=predictions["logits"].device))
        return self._loss(predictions, torch.tensor(targets["labels"], device=predictions.device))

    def loss2(self, predictions:torch.tensor, targets:list):
        return self._loss2(predictions, torch.tensor(targets, device=predictions.device))


    def encode_labels(self, labels):
        if self.hparams.tune_sp6_labels or self.hparams.train_enc_dec_sp6:
            all_lbls = []
            all_padded_lbls = []
            seqs_lengths = []
            global_labels = []
            max_len = 0
            for sample in labels:
                all_lbls.append(sample['target'])
                max_len = len(sample['target']) if len(sample['target']) > max_len else max_len
            for lbl_seq in all_lbls:
                if hparams.use_glbl_labels:
                    lbl_seq_, glbl_lbl = lbl_seq
                    padded_lbl = lbl_seq_.copy()
                    global_labels.append(glbl_lbl)
                    seqs_lengths.append(len(lbl_seq_))
                else:
                    padded_lbl = lbl_seq.copy()
                    seqs_lengths.append(len(lbl_seq))
                all_padded_lbls.append(padded_lbl)
            if self.hparams.train_enc_dec_sp6:
                if hparams.use_glbl_labels:
                    return all_padded_lbls, seqs_lengths, global_labels
                return all_padded_lbls, seqs_lengths
            return all_padded_lbls
        vocab = {"L": 0, "A": 1, "G": 2, "V": 3, "E": 4, "S": 5,
                 "I": 6, "K": 7, "R": 8, "D": 9, "T": 10, "P": 11, "N": 12, "Q": 13, "F": 14, "Y": 15,
                 "M": 16, "H": 17, "C": 18, "W": 19, "X": 20, "U": 21, "B": 22, "Z": 23, "O": 24}
        bs = len(labels[0])
        all_labels = []
        for i in range(bs):
            all_labels.append([vocab[l[i]] for l in labels])
        return all_labels

    def prepare_sample(self, sample: list, prepare_target: bool = True) -> (dict, dict):
        """
        Function that prepares a sample to input the model.
        :param sample: list of dictionaries.

        Returns:
            - dictionary with the expected model inputs.
            - dictionary with the expected target labels.
        """

        if self.hparams.train_enc_dec_sp6:
            if hparams.use_glbl_labels:
                target, seq_lengths, glbl_labels = self.encode_labels(sample)
            else:
                target, seq_lengths = self.encode_labels(sample)
        elif self.hparams.tune_sp6_labels:
            target = self.encode_labels(sample)
        sample = collate_tensors(sample)
        inputs = self.tokenizer.batch_encode_plus(sample["seq"],
                                                  add_special_tokens=self.hparams.special_tokens,
                                                  padding=True,
                                                  truncation=True,
                                                  max_length=self.hparams.max_length)
        if self.hparams.train_enc_dec_sp6:
            if hparams.use_glbl_labels:
                return inputs, target, seq_lengths, glbl_labels
            return inputs, target, seq_lengths
        if self.hparams.tune_sp6_labels:
            return inputs, target
        if not prepare_target:
            return inputs, {}

        # Prepare target:

        try:
            targets = {"labels": self.encode_labels(sample["target"])}
            if self.hparams.tune_epitope_specificity:
                return inputs, targets
            return inputs, targets, sample["target_pos"]
        except RuntimeError:
            raise Exception("Label encoder found an unknown label.")

    def training_step(self, batch: tuple, batch_nb: int, *args, **kwargs) -> dict:
        # self.evaluate(self.classification_head, self.lbl2ind_dict, run_name=self.hparams.run_name, partitions=hparams.train_folds,
        #               form_sp_reg_data=hparams.use_glbl_labels, simplified=hparams.use_glbl_labels, very_simplified=hparams.use_glbl_labels, glbl_lbl_2ind=self.glbl_lbl2ind ,)
        """
        Runs one training step. This usually consists in the forward function followed
            by the loss function.

        :param batch: The output of your dataloader.
        :param batch_nb: Integer displaying which batch this is
        Returns:
            - dictionary containing the loss and the metrics to be added to the lightning logger.
        """
        # inputs, targets = batch
        self.classification_head.train()
        if self.train_BFD:
            self.ProtBertBFD.train()

        if self.hparams.tune_epitope_specificity:
            inputs, targets = batch
            model_out = self.forward(**inputs)
            loss_val = self.loss(model_out, targets)
            tqdm_dict = {"train_loss": loss_val}
            output = OrderedDict(
                {"loss": loss_val, "progress_bar": tqdm_dict, "log": tqdm_dict})
            # can also return just a scalar instead of a dict (return loss_val)
            return output
        if self.hparams.tune_sp6_labels:
            inputs, targets = batch
            model_out = self.forward(**inputs, v=True)
            loss_val = self.loss(model_out.reshape(-1, len(self.lbl2ind_dict.keys())),
                                 {"labels": list(np.array(targets).reshape(-1))})
            tqdm_dict = {"train_loss": loss_val}
            output = OrderedDict(
                {"loss": loss_val, "progress_bar": tqdm_dict, "log": tqdm_dict})
            # can also return just a scalar instead of a dict (return loss_val)
            return output
        elif self.hparams.train_enc_dec_sp6:
            if hparams.use_glbl_labels:
                inputs, targets, seq_lengths, glbl_labels = batch
            else:
                inputs, targets, seq_lengths = batch
            inputs['targets'] = targets
            inputs['seq_lengths'] = seq_lengths
            if hparams.use_glbl_labels:
                model_out, glbl_out = self.forward(**inputs, v=True)
            else:
                model_out = self.forward(**inputs)

            eos_token_targets = []
            max_len = max(seq_lengths)
            for t, sl in zip(targets, seq_lengths):
                eos_token_targets.append(t)
                eos_token_targets[-1].append(self.lbl2ind_dict['ES'])
                eos_token_targets[-1].extend([self.lbl2ind_dict['PD']] * (max_len - sl))
            # ind2lbl = {v:k for k,v in self.lbl2ind_dict.items()}
            # for eo_t_t in eos_token_targets:
            #     print("".join(ind2lbl[e] for e in eo_t_t))
            # print(eos_token_targets)
            # print([len(a) for a in list(np.array(eos_token_targets))])
            # print("model_out.shape", model_out.shape)
            # print(inputs['input_ids'])
            # print(model_out[-1, 0, :], model_out.shape)
            # exit(1)
            # print(model_out.shape)
            loss_val = self.loss(model_out.transpose(1, 0).reshape(-1, len(self.lbl2ind_dict.keys())),
                                 {"labels": list(np.array(eos_token_targets).reshape(-1))})
            if hparams.use_glbl_labels:
                loss_glbl = self.loss2(glbl_out, glbl_labels)
                loss_val += loss_glbl
            tqdm_dict = {"train_loss": loss_val}
            output = OrderedDict(
                {"loss": loss_val, "progress_bar": tqdm_dict, "log": tqdm_dict})
            # can also return just a scalar instead of a dict (return loss_val)
            return output
        inputs, targets, target_positions = batch
        all_targets = []
        for tl in targets["labels"]:
            all_targets.extend(tl)
        targets["labels"] = all_targets

        bs = len(target_positions[0])
        all_target_pos = []
        for i in range(bs):
            all_target_pos.append([tp_[i] for tp_ in target_positions])
        target_positions = all_target_pos
        inputs["target_positions"] = target_positions

        model_out = self.forward(**inputs)
        loss_val = self.loss(model_out, targets)

        tqdm_dict = {"train_loss": loss_val}
        output = OrderedDict(
            {"loss": loss_val, "progress_bar": tqdm_dict, "log": tqdm_dict})
        # can also return just a scalar instead of a dict (return loss_val)
        return output

    def validation_step(self, batch: tuple, batch_nb: int, *args, **kwargs) -> dict:
        """ Similar to the training step but with the model in eval mode.
        Returns:
            - dictionary passed to the validation_end function.
        """
        self.ProtBertBFD.eval()
        self.classification_head.eval()
        if self.hparams.tune_epitope_specificity:
            inputs, targets = batch
            model_out = self.forward(**inputs)
            loss_val = self.loss(model_out, targets)
            y = targets["labels"]
            # y_hat = model_out["logits"]
            y_hat = model_out
            labels_hat = torch.argmax(y_hat, dim=1)
            # labels_hat = y_hat
            val_acc = self.metric_acc(labels_hat, y)

            output = OrderedDict({"val_loss": loss_val, "val_acc": val_acc, "mcc":10})
            return output
        elif self.hparams.tune_sp6_labels:
            inputs, targets = batch
            model_out = self.forward(**inputs)
            loss_val = self.loss(model_out.reshape(-1, len(self.lbl2ind_dict.keys())),
                                 {"labels": list(np.array(targets).reshape(-1))})
            y = targets
            y_hat = model_out
            labels_hat = torch.argmax(y_hat, dim=-1)
            val_acc = self.metric_acc(labels_hat.reshape(-1), list(np.array(y).reshape(-1)))
            output = OrderedDict({"val_loss": loss_val, "val_acc": val_acc, })
            return output
        elif self.hparams.train_enc_dec_sp6:

            if hparams.use_glbl_labels:
                inputs, targets, seq_lengths, glbl_labels = batch
            else:
                inputs, targets, seq_lengths = batch
            inputs['targets'] = targets
            inputs['seq_lengths'] = seq_lengths
            if hparams.use_glbl_labels:
                model_out, glbl_out = self.forward(**inputs)
            else:
                model_out = self.forward(**inputs)
            eos_token_targets = []
            max_len = max(seq_lengths)
            for t, sl in zip(targets, seq_lengths):
                eos_token_targets.append(t)
                eos_token_targets[-1].append(self.lbl2ind_dict['ES'])
                eos_token_targets[-1].extend([self.lbl2ind_dict['PD']] * (max_len - sl))
            loss_val = self.loss(model_out.permute(1, 0, 2).reshape(-1, len(self.lbl2ind_dict.keys())),
                                 {"labels": list(np.array(eos_token_targets).reshape(-1))})
            if hparams.use_glbl_labels:
                loss_glbl = self.loss2(glbl_out, glbl_labels)
                loss_val += loss_glbl
            y = eos_token_targets
            y_hat = model_out
            labels_hat = torch.argmax(y_hat, dim=-1)
            val_acc = self.metric_acc(labels_hat.reshape(-1), list(np.array(y).reshape(-1)))
            if self.hparams.validate_on_mcc:
                if self.hparams.use_glbl_labels:
                    inputs, tgt, seq_lengths, glbl_lbls = batch
                else:
                    inputs, tgt, seq_lengths = batch
                all_lbls, all_predicted_lbls = [], []
                # print("Number of sequences tested: {}".format(ind * test_batch_size))
                ind2lbl = {v: k for k, v in self.lbl2ind_dict.items()}
                ind2glbl_lbl = {v: k for k, v in self.glbl_lbl2ind.items()}

                predicted_tokens, probs, sp_probs, all_sp_probs, all_seq_sp_logits, sp_type_probs = \
                    self.translate(self.classification_head, inputs, self.lbl2ind_dict['BS'], self.lbl2ind_dict, tgt=None, use_beams_search=False,
                              form_sp_reg_data=self.hparams.use_glbl_labels, second_model=None, test_only_cs=False,
                              glbl_lbls=self.glbl_lbl2ind if self.hparams.use_glbl_labels else None)
                tgt, predicted_tokens = np.array(tgt), np.array(predicted_tokens)
                glbl_lbls_pred = torch.argmax(sp_type_probs, dim=1).cpu().numpy()
                tp, tn, fp, fn = 0, 0, 0, 0
                if not self.hparams.use_glbl_labels:
                    min_lens = [min(len(tgt_), len(prd_), 70) for tgt_, prd_ in zip(tgt, predicted_tokens)]
                    all_tgts, all_pred_tkns = [], []
                    for i in range(len(tgt)):
                        all_tgts.extend(tgt[i][:min_lens[i]])
                        all_pred_tkns.extend(predicted_tokens[i][:min_lens[i]])
                    all_tgts = np.array(all_tgts)
                    all_pred_tkns = np.array(all_pred_tkns)
                    for pred, target in zip(all_pred_tkns, all_tgts):
                        if pred == target and target in [0, 1, 4, 6, 7]:
                            tp += 1
                        elif pred == target and target not in [0, 1, 4, 6, 7]:
                            tn += 1
                        elif pred != target and target in [0, 1, 4, 6, 7]:
                            fn += 1
                        else:
                            fp += 1
                else:
                    for target, pred, glbl_l_t, glbl_lbl_p in zip(tgt, predicted_tokens, glbl_lbls, glbl_lbls_pred):
                        min_len = min(70, len(pred), len(target))
                        for p, t in zip(pred[:min_len], target[:min_len]):
                            if p == t and t == 0:
                                if glbl_l_t == glbl_lbl_p:
                                    tp +=1
                                else:
                                    fn += 1
                            elif p == t and t !=0:
                                tn += 1
                            elif p != t and t == 0:
                                fn += 1
                            elif p != t and t != 0:
                                fp += 1
                output = OrderedDict({"val_loss": loss_val, "val_acc": val_acc, "tp":tp, "tn":tn, "fn":fn, "fp":fp})
                return output
            output = OrderedDict({"val_loss": loss_val, "val_acc": val_acc, })
            return output
        inputs, targets, target_positions = batch
        all_targets = []
        for tl in targets["labels"]:
            all_targets.extend(tl)
        targets["labels"] = all_targets
        bs = len(target_positions[0])
        all_target_pos = []
        for i in range(bs):
            all_target_pos.append([tp_[i] for tp_ in target_positions])
        target_positions = all_target_pos
        inputs["target_positions"] = target_positions
        model_out = self.forward(**inputs)
        loss_val = self.loss(model_out, targets)
        y = targets["labels"]
        # y_hat = model_out["logits"]
        y_hat = model_out
        labels_hat = torch.argmax(y_hat, dim=1)
        # labels_hat = y_hat
        val_acc = self.metric_acc(labels_hat, y)
        output = OrderedDict({"val_loss": loss_val, "val_acc": val_acc, })
        return output

    def validation_epoch_end(self, outputs: list) -> dict:
        """ Function that takes as input a list of dictionaries returned by the validation_step
        function and measures the model performance accross the entire validation set.

        Returns:
            - Dictionary with metrics to be added to the lightning logger.
        """
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        val_acc_mean = torch.stack([x['val_acc'] for x in outputs]).mean()
        if self.hparams.validate_on_mcc:
            tp, tn, fn, fp = sum(x['tp'] for x in outputs), sum(x['tn'] for x in outputs), sum(x['fn'] for x in outputs), sum(x['fp'] for x in outputs)
            if tp == 0 or tn == 0:
                mcc_mean = -1
            else:
                mcc_mean = (tp * tn - fp * fn)/(np.sqrt( (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn) ))

            tqdm_dict = {"val_loss": val_loss_mean, "val_acc": val_acc_mean, "mcc_mean": torch.tensor(mcc_mean)}
            result = {
                "progress_bar": tqdm_dict,
                "log": tqdm_dict,
                "val_loss": val_loss_mean,
            }
        else:

            tqdm_dict = {"val_loss": val_loss_mean, "val_acc": val_acc_mean}
            result = {
                "progress_bar": tqdm_dict,
                "log": tqdm_dict,
                "val_loss": val_loss_mean,
            }
        return result

    def greedy_decode(self, model, src, start_symbol, lbl2ind, tgt=None, form_sp_reg_data=False, second_model=None,
                      test_only_cs=False, glbl_lbls=None):
        ind2glbl_lbl = {0: 'NO_SP', 1: 'SP', 2: 'TATLIPO', 3: 'LIPO', 4: 'TAT', 5: 'PILIN'}
        ind2lbl = {v: k for k, v in lbl2ind.items()}
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        src = src
        inp_seqs = []
        for inp in src['input_ids']:
            inp_seqs.append("".join([self.aaind2lblvocab[i_] for i_ in inp]).replace("[PAD]", ""))
        seq_lens = [len(src_) for src_ in src['input_ids']]
        sp_probs = []
        sp_logits = []
        all_seq_sp_probs = []
        all_seq_sp_logits = []
        # used for glbl label version 2
        all_outs = []
        all_outs_2nd_mdl = []
        glbl_labels = None
        with torch.no_grad():
            input_ids = torch.tensor(src['input_ids'], device=self.device)
            attention_mask = torch.tensor(src['attention_mask'], device=self.device)
            memory_bfd = self.ProtBertBFD(input_ids=input_ids, attention_mask=attention_mask)[0]
            memory = self.classification_head.encode(memory_bfd, inp_seqs=inp_seqs)
            memory_2nd_mdl = None
            if self.classification_head.glbl_lbl_version == 3:
                if test_only_cs:
                    batch_size = len(src)
                    glbl_labels = torch.zeros(batch_size, 6)
                    glbl_labels[list(range(batch_size)), glbl_lbls] = 1
                    _, glbl_preds = torch.max(glbl_labels, dim=1)
                elif form_sp_reg_data:
                    glbl_labels = model.get_v3_glbl_lbls(memory_bfd, inp_seqs=inp_seqs)
                    _, glbl_preds = torch.max(glbl_labels, dim=1)

        # ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
        ys = []

        if not ys:
            for _ in range(len(src['input_ids'])):
                ys.append([])
        start_ind = 0
        # if below condition, then global labels are computed with a separate model. This also affects the first label pred
        # of the model predicting the sequence label. Because of this, compute the first predictions first, and take care
        # of the glbl label model and sequence-model consistency (e.g. one predicts SP other NO-SP - take care of that)
        if glbl_labels is not None:
            tgt_mask = (self.generate_square_subsequent_mask(1))
            out = model.decode(ys, memory.to(device), tgt_mask.to(device))
            out = out.transpose(0, 1)
            prob = model.generator(out[:, -1])
            _, next_words = torch.max(prob, dim=1)
            next_word = [nw.item() for nw in next_words]
            current_ys = []
            # ordered indices of no-sp aa labels O, M, I
            ordered_no_sp = [lbl2ind['O'], lbl2ind['M'], lbl2ind['I']]
            for bach_ind in range(len(src['input_ids'])):
                if ind2glbl_lbl[glbl_preds[bach_ind].item()] == "NO_SP" and ind2lbl[next_word[bach_ind]] == "S":
                    max_no_sp = np.argmax([prob[bach_ind][lbl2ind['O']].item(), prob[bach_ind][lbl2ind['M']].item(),
                                           prob[bach_ind][lbl2ind['I']].item()])
                    current_ys.append(ys[bach_ind])
                    current_ys[-1].append(ordered_no_sp[max_no_sp])
                elif ind2glbl_lbl[glbl_preds[bach_ind].item()] in ['SP', 'TATLIPO', 'LIPO', 'TAT', 'PILIN'] \
                        and ind2lbl[next_word[bach_ind]] != "S":
                    current_ys.append(ys[bach_ind])
                    current_ys[-1].append(lbl2ind["S"])
                else:
                    current_ys.append(ys[bach_ind])
                    current_ys[-1].append(next_word[bach_ind])
            ys = current_ys
            start_ind = 1
        model.eval()
        if form_sp_reg_data:
            model.glbl_generator.eval()
        all_probs = []
        for i in range(start_ind, max(seq_lens) + 1):
            with torch.no_grad():
                tgt_mask = (self.generate_square_subsequent_mask(len(ys[0]) + 1))
                if second_model is not None:
                    out_2nd_mdl = second_model.decode(ys, memory_2nd_mdl.to(device), tgt_mask.to(device))
                    out_2nd_mdl = out_2nd_mdl.transpose(0, 1)
                    prob_2nd_mdl = second_model.generator(out_2nd_mdl[:, -1])
                    all_outs_2nd_mdl.append(out_2nd_mdl[:, -1])
                out = model.decode(ys, memory.to(device), tgt_mask.to(device))
                out = out.transpose(0, 1)
                prob = model.generator(out[:, -1])
                all_outs.append(out[:, -1])

            if i == 0 and not form_sp_reg_data:
                # extract the sp-presence probabilities
                glbl_labels = torch.nn.functional.softmax(prob, dim=-1)
                sp_probs = [sp_prb.item() for sp_prb in torch.nn.functional.softmax(prob, dim=-1)[:, lbl2ind['S']]]
                all_seq_sp_probs = [[sp_prob.item()] for sp_prob in
                                    torch.nn.functional.softmax(prob, dim=-1)[:, lbl2ind['S']]]
                all_seq_sp_logits = [[sp_prob.item()] for sp_prob in prob[:, lbl2ind['S']]]
            elif not form_sp_reg_data:
                # used to update the sequences of probabilities
                softmax_probs = torch.nn.functional.softmax(prob, dim=-1)
                next_sp_probs = softmax_probs[:, lbl2ind['S']]
                next_sp_logits = prob[:, lbl2ind['S']]
                for seq_prb_ind in range(len(all_seq_sp_probs)):
                    all_seq_sp_probs[seq_prb_ind].append(next_sp_probs[seq_prb_ind].item())
                    all_seq_sp_logits[seq_prb_ind].append(next_sp_logits[seq_prb_ind].item())
            all_probs.append(prob)
            if second_model is not None:
                probs_fm, next_words_fm = torch.max(torch.nn.functional.softmax(prob, dim=-1), dim=1)
                probs_sm, next_words_sm = torch.max(torch.nn.functional.softmax(prob_2nd_mdl, dim=-1), dim=1)
                all_probs_mdls = torch.stack([probs_fm, probs_sm])
                all_next_w_mdls = torch.stack([next_words_fm, next_words_sm])
                if i == 0:
                    _, inds = torch.max(all_probs_mdls, dim=0)
                next_words = all_next_w_mdls[inds, torch.tensor(list(range(inds.shape[0])))]
            else:
                _, next_words = torch.max(prob, dim=1)
            next_word = [nw.item() for nw in next_words]
            current_ys = []
            for bach_ind in range(len(src['input_ids'])):
                current_ys.append(ys[bach_ind])
                current_ys[-1].append(next_word[bach_ind])
            ys = current_ys
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if form_sp_reg_data:
            if model.glbl_lbl_version == 2 and model.use_glbl_lbls:
                if model.version2_agregation == "max":
                    glbl_labels = model.glbl_generator(torch.max(torch.stack(all_outs).transpose(0, 1), dim=1)[0])
                    if second_model is not None:
                        glbl_labels = torch.stack([torch.nn.functional.softmax(glbl_labels, dim=1),
                                                   torch.nn.functional.softmax(second_model.glbl_generator(
                                                       torch.max(torch.stack(all_outs_2nd_mdl).transpose(0, 1), dim=1)[
                                                           0]),
                                                       dim=1)])
                        glbl_labels = glbl_labels[inds, torch.tensor(list(range(inds.shape[0]))), :]


                elif model.version2_agregation == "avg":
                    glbl_labels = model.glbl_generator(torch.mean(torch.stack(all_outs).transpose(0, 1), dim=1))
                    if second_model is not None:
                        glbl_labels = torch.nn.functional.softmax(glbl_labels) + \
                                      torch.nn.functional.softmax(second_model.glbl_generator(
                                          torch.mean(torch.stack(all_outs_2nd_mdl).transpose(0, 1), dim=1)), -1)
            elif model.glbl_lbl_version == 1 and model.use_glbl_lbls:
                glbl_labels = model.glbl_generator(memory.transpose(0, 1)[:, 1, :])
                if second_model is not None:
                    glbl_labels = torch.nn.functional.softmax(glbl_labels, dim=-1) + \
                                  torch.nn.functional.softmax(
                                      second_model.glbl_generator(memory_2nd_mdl.transpose(0, 1)[:, 1, :]), dim=-1)
            elif model.glbl_lbl_version != 3:
                glbl_labels = model.glbl_generator(
                    torch.mean(torch.sigmoid(torch.stack(all_probs)).transpose(0, 1), dim=1))
                if second_model is not None:
                    glbl_labels = torch.nn.functional.softmax(glbl_labels, dim=-1) + \
                                  second_model.glbl_generator(
                                      torch.mean(torch.sigmoid(torch.stack(all_probs)).transpose(0, 1), dim=1), dim=-1)
            return ys, torch.stack(all_probs).transpose(0, 1), sp_probs, all_seq_sp_probs, all_seq_sp_logits, \
                   glbl_labels
        return ys, torch.stack(all_probs).transpose(0, 1), sp_probs, all_seq_sp_probs, all_seq_sp_logits, glbl_labels

    def evaluate(self, model, lbl2ind, run_name="", test_batch_size=50, partitions=[0, 1], sets=["train"], epoch=-1,
                 dataset_loader=None, use_beams_search=False, form_sp_reg_data=False, simplified=False,
                 second_model=None,
                 very_simplified=False, test_only_cs=False, glbl_lbl_2ind=None, account_lipos=False,
                 tuned_bert_embs_prefix=""):
        if glbl_lbl_2ind is not None:
            ind2glbl_lbl = {v: k for k, v in glbl_lbl_2ind.items()}
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=lbl2ind["PD"])
        eval_dict = {}
        sp_type_dict = {}
        seqs2probs = {}
        model.eval()
        self.ProtBertBFD.eval()
        if second_model is not None:
            second_model.eval()
        val_or_test = "test" if len(sets) == 2 else "validation"
        dataset = SP6TuningDataset(self.hparams.relative_data_path, self.hparams.dev_csv,
                                   use_glbl_lbls=self.hparams.use_glbl_labels)
        dataset_loader =  DataLoader(
                            dataset=dataset,
                            batch_size=self.hparams.batch_size,
                            collate_fn=self.prepare_sample,
                            num_workers=self.hparams.loader_workers,
                            shuffle=False)


        ind2lbl = {v: k for k, v in lbl2ind.items()}
        total_loss = 0
        aaind2lblvocab = {v:k for k, v in self.tokenizer.get_vocab().items()}

        # for ind, (src, tgt, _, glbl_lbls) in tqdm(enumerate(dataset_loader), "Epoch {} {}".format(epoch, val_or_test),
        #                                           total=len(dataset_loader)):
        for ind, test_datapoints in tqdm(enumerate(dataset_loader),
                                                  "Epoch {} {}".format(epoch, val_or_test),
                                                  total=len(dataset_loader)):
            if form_sp_reg_data:
                inputs, tgt, seq_lengths, glbl_lbls = test_datapoints
            else:
                inputs, tgt, seq_lengths = test_datapoints

            # print("Number of sequences tested: {}".format(ind * test_batch_size))
            predicted_tokens, probs, sp_probs, all_sp_probs, all_seq_sp_logits, sp_type_probs = \
                self.translate(model, inputs, lbl2ind['BS'], lbl2ind, tgt=None, use_beams_search=use_beams_search,
                          form_sp_reg_data=form_sp_reg_data, second_model=second_model, test_only_cs=test_only_cs,
                          glbl_lbls=None)
            src = []
            for inds in inputs['input_ids']:

                src.append("".join([aaind2lblvocab[i_] for i_ in inds]).replace("[PAD]", ""))
            true_targets = padd_add_eos_tkn(tgt, lbl2ind)
            # if not use_beams_search:
            #     total_loss += loss_fn(probs.reshape(-1, 10), true_targets.reshape(-1)).item()
            for s, t, pt, sp_type in zip(src, tgt, predicted_tokens, sp_type_probs):
                predicted_lbls = "".join([ind2lbl[i] for i in pt])
                if account_lipos:
                    predicted_lbls, sp_type = clean_sec_sp2_preds(s, predicted_lbls, sp_type, ind2glbl_lbl)
                sp_type_dict[s] = torch.argmax(sp_type).item()
                if form_sp_reg_data:
                    new_predicted_lbls = modify_sp_subregion_preds(predicted_lbls, sp_type)
                    predicted_lbls = new_predicted_lbls
                eval_dict[s] = predicted_lbls[:len(t)]
            if sp_probs is not None:
                for s, sp_prob, all_sp_probs, all_sp_logits in zip(src, sp_probs, all_sp_probs, all_seq_sp_logits):
                    seqs2probs[s] = (sp_prob, all_sp_probs, all_sp_logits)
        pickle.dump(eval_dict, open(run_name + ".bin", "wb"))
        pickle.dump(sp_type_dict, open(run_name + "_sptype.bin", "wb"))
        sp_pred_mccs, sp_pred_mccs2, lipo_pred_mccs, lipo_pred_mccs2, tat_pred_mccs, tat_pred_mccs2, \
        all_recalls_lipo, all_precisions_lipo, all_recalls_tat, all_precisions_tat, all_f1_scores_lipo, all_f1_scores_tat, \
        all_recalls, all_precisions, total_positives, false_positives, predictions, all_f1_scores, sptype_f1 = \
            get_cs_and_sp_pred_results(filename=self.hparams.run_name + ".bin", v=False, return_everything=True,
                                       return_class_prec_rec=True)
        all_recalls, all_precisions, total_positives = list(np.array(all_recalls).flatten()), \
                                                       list(np.array(all_precisions).flatten()), list(
            np.array(total_positives).flatten())
        log_and_print_mcc_and_cs_results(sp_pred_mccs, all_recalls, all_precisions, test_on="VALIDATION", ep=self.e,
                                         all_f1_scores=all_f1_scores, sptype_f1=sptype_f1)
        model.train()
        if sp_probs is not None and len(sets) > 1:
            # retrieve the dictionary of calibration only for the test set (not for validation) - for now it doesn't
            # make sense to do prob calibration since like 98% of predictions have >0.99 and are correct. See with weight decay
            pickle.dump(seqs2probs, open(run_name + "_sp_probs.bin", "wb"))
        return total_loss / len(dataset_loader)

    def translate(self, model, src, bos_id, lbl2ind, tgt=None, use_beams_search=False,
                  form_sp_reg_data=False, second_model=None, test_only_cs=False, glbl_lbls=None):
        tgt_tokens, probs, sp_probs, \
        all_sp_probs, all_seq_sp_logits, sp_type_probs = self.greedy_decode(model, src, start_symbol=bos_id,
                                                                    lbl2ind=lbl2ind,
                                                                    tgt=tgt,
                                                                    form_sp_reg_data=form_sp_reg_data,
                                                                    second_model=second_model,
                                                                    test_only_cs=test_only_cs,
                                                                    glbl_lbls=glbl_lbls)
        sp_type_probs_ = []
        if not form_sp_reg_data:
            for sp_prbs in sp_type_probs:
                # ind2glbl_lbl = {0: 'NO_SP', 1: 'SP', 2: 'TATLIPO', 3: 'LIPO', 4: 'TAT', 5: 'PILIN'}
                sp, tatlipo, lipo, tat, pilin = sp_prbs[self.lbl2ind_dict['S']].item(), sp_prbs[self.lbl2ind_dict['W']].item(), \
                                                sp_prbs[self.lbl2ind_dict['L']].item(), sp_prbs[self.lbl2ind_dict['T']].item(), \
                                                sp_prbs[self.lbl2ind_dict['P']].item()
                no_sp = 1 - (sp + tatlipo + lipo + tat + pilin)
                sp_type_probs_.append(torch.tensor([no_sp, sp, tatlipo, lipo, tat, pilin]))
            sp_type_probs = torch.stack(sp_type_probs_)
        return tgt_tokens, probs, sp_probs, \
               all_sp_probs, all_seq_sp_logits, sp_type_probs

    def test_step(self, batch: tuple, batch_nb: int, *args, **kwargs) -> dict:
        """ Similar to the training step but with the model in eval mode.
        Returns:
            - dictionary passed to the validation_end function.
        """
        inputs, targets = batch
        model_out = self.forward(**inputs)
        loss_test = self.loss(model_out, targets)

        y = targets["labels"]
        # y_hat = model_out["logits"]
        y_hat = model_out

        labels_hat = torch.argmax(y_hat, dim=1)
        # labels_hat = y_hat
        test_acc = self.metric_acc(labels_hat, y)

        output = OrderedDict({"test_loss": loss_test, "test_acc": test_acc, })
        return output

    def test_epoch_end(self, outputs: list) -> dict:
        """ Function that takes as input a list of dictionaries returned by the validation_step
        function and measures the model performance accross the entire validation set.

        Returns:
            - Dictionary with metrics to be added to the lightning logger.
        """
        test_loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()
        test_acc_mean = torch.stack([x['test_acc'] for x in outputs]).mean()

        tqdm_dict = {"test_loss": test_loss_mean, "test_acc": test_acc_mean}
        result = {
            "progress_bar": tqdm_dict,
            "log": tqdm_dict,
            "test_loss": test_loss_mean,
        }
        return result

    def configure_optimizers(self):
        """ Sets different Learning rates for different parameter groups. """
        parameters = [
            {"params": self.classification_head.parameters()},
            {
                "params": self.ProtBertBFD.parameters(),
                "lr": self.hparams.encoder_learning_rate,
            },
        ]
        # optimizer = Lamb(parameters, lr=self.hparams.learning_rate, weight_decay=0.01)
        optimizer = optim.Adam(parameters, lr=self.hparams.learning_rate,  eps=1e-9,)
        return [optimizer], []

    def on_epoch_end(self):
        """ Pytorch lightning hook """
        self.evaluate(self.classification_head, self.lbl2ind_dict, run_name=self.hparams.run_name, partitions=hparams.train_folds,
                      form_sp_reg_data=hparams.use_glbl_labels, simplified=hparams.use_glbl_labels, very_simplified=hparams.use_glbl_labels, glbl_lbl_2ind=self.glbl_lbl2ind ,)
        self.e += 1
        if self.current_epoch + 1 >= self.nr_frozen_epochs:
            self.unfreeze_encoder()

    def __retrieve_dataset(self, train=True, val=True, test=True):
        """ Retrieves task specific dataset """
        if self.hparams.tune_epitope_specificity:
            if train:
                return EpitopeClassifierDataset(hparams.relative_data_path, hparams.train_csv)
            elif val:
                return EpitopeClassifierDataset(hparams.relative_data_path, hparams.dev_csv)
            elif test:
                return EpitopeClassifierDataset(hparams.relative_data_path, hparams.test_csv)
            else:
                print('Incorrect dataset split')
        elif self.hparams.tune_sp6_labels or self.hparams.train_enc_dec_sp6:
            if train:
                return SP6TuningDataset(hparams.relative_data_path, hparams.train_csv, hparams.use_glbl_labels)
            elif val:
                return SP6TuningDataset(hparams.relative_data_path, hparams.dev_csv,hparams.use_glbl_labels)
            elif test:
                return SP6TuningDataset(hparams.relative_data_path, hparams.test_csv,hparams.use_glbl_labels)
        else:
            if train:
                return BertDataset(hparams.train_csv, hparams.special_tokens, hparams.relative_data_path)
            elif val:
                return BertDataset(hparams.dev_csv, hparams.special_tokens, hparams.relative_data_path)
            elif test:
                return BertDataset(hparams.test_csv, hparams.special_tokens, hparams.relative_data_path)
            else:
                print('Incorrect dataset split')

    def train_dataloader(self) -> DataLoader:
        """ Function that loads the train set. """
        self._train_dataset = self.__retrieve_dataset(val=False, test=False)
        return DataLoader(
            dataset=self._train_dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=self.prepare_sample,
            num_workers=self.hparams.loader_workers,
        )

    def val_dataloader(self) -> DataLoader:
        """ Function that loads the validation set. """
        self._dev_dataset = self.__retrieve_dataset(train=False, test=False)
        return DataLoader(
            dataset=self._dev_dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=self.prepare_sample,
            num_workers=self.hparams.loader_workers,
        )

    def test_dataloader(self) -> DataLoader:
        """ Function that loads the validation set. """
        self._test_dataset = self.__retrieve_dataset(train=False, val=False)
        return DataLoader(
            dataset=self._test_dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=self.prepare_sample,
            num_workers=self.hparams.loader_workers,
        )

    @classmethod
    def add_model_specific_args(
            cls, parser: HyperOptArgumentParser
    ) -> HyperOptArgumentParser:
        """ Parser for Estimator specific arguments/hyperparameters.
        :param parser: HyperOptArgumentParser obj
        Returns:
            - updated parser
        """
        parser.opt_list(
            "--max_length",
            default=1536,
            type=int,
            help="Maximum sequence length.",
        )
        parser.add_argument(
            "--encoder_learning_rate",
            default=1e-6,
            type=float,
            help="Encoder specific learning rate.",
        )
        parser.add_argument(
            "--learning_rate",
            default=1e-5,
            type=float,
            help="Classification head learning rate.",
        )
        parser.opt_list(
            "--nr_frozen_epochs",
            default=1,
            type=int,
            help="Number of epochs we want to keep the encoder model frozen.",
            tunable=True,
        )
        # Data Args:
        parser.add_argument(
            "--label_set",
            default="M,S",
            type=str,
            help="Classification labels set.",
        )
        parser.add_argument(
            "--train_csv",
            default="tcr_seqs_train_df.csv",
            type=str,
            help="Path to the file containing the train data.",
        )
        parser.add_argument(
            "--dev_csv",
            default="tcr_seqs_dev_df.csv",
            type=str,
            help="Path to the file containing the dev data.",
        )
        parser.add_argument(
            "--test_csv",
            default="tcr_seqs_test_df.csv",
            type=str,
            help="Path to the file containing the dev data.",
        )
        parser.add_argument(
            "--loader_workers",
            default=8,
            type=int,
            help="How many subprocesses to use for data loading. 0 means that \
                the data will be loaded in the main process.",
        )
        parser.add_argument(
            "--gradient_checkpointing",
            default=False,
            action="store_true",
            help="Enable or disable gradient checkpointing which use the cpu memory \
                with the gpu memory to store the model.",
        )
        return parser


def setup_testube_logger(save_dir="experiments") -> TestTubeLogger:
    """ Function that sets the TestTubeLogger to be used. """
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y--%H-%M-%S")
    if os.path.exists("/scratch/work/dumitra1/"):
        save_dir = "/scratch/work/dumitra1/sp_data/experiments_sp6tuning/"
    else:
        save_dir = "experiments/"
    return TestTubeLogger(
        save_dir=save_dir,
        version=dt_string,
        name="lightning_logs",
    )


def parse_arguments_and_retrieve_logger(save_dir="experiments"):
    """
        Function for parsing all arguments
    """
    logger = setup_testube_logger(save_dir=save_dir)
    parser = HyperOptArgumentParser(
        strategy="random_search",
        description="Minimalist ProtBERT Classifier",
        add_help=True,
    )
    parser.add_argument("--seed", type=int, default=3, help="Training seed.")
    parser.add_argument(
        "--save_top_k",
        default=1,
        type=int,
        help="The best k models according to the quantity monitored will be saved.",
    )
    parser.add_argument(
        "--create_data",
        default=False,
        action="store_true",
        help="Create data for bert fine-tuning out of emerson-long and tcrb files",
    )
    parser.add_argument(
        "--special_tokens",
        default=False,
        action="store_true",
        help="Tune the ProtBert ot the epitope classification task"
    )
    # Early Stopping
    parser.add_argument(
        "--monitor", default="val_acc", type=str, help="Quantity to monitor."
    )
    parser.add_argument(
        "--metric_mode",
        default="max",
        type=str,
        help="If we want to min/max the monitored quantity.",
        choices=["auto", "min", "max"],
    )
    parser.add_argument(
        "--patience",
        default=2,
        type=int,
        help=(
            "Number of epochs with no improvement "
            "after which training will be stopped."
        ),
    )
    parser.add_argument(
        "--min_epochs",
        default=1,
        type=int,
        help="Limits training to a minimum number of epochs",
    )
    parser.add_argument(
        "--max_epochs",
        default=100,
        type=int,
        help="Limits training to a max number number of epochs",
    )

    # Batching
    parser.add_argument(
        "--batch_size", default=1, type=int, help="Batch size to be used."
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        default=64,
        type=int,
        help=(
            "Accumulated gradients runs K small batches of size N before "
            "doing a backwards pass."
        ),
    )
    parser.add_argument(
        "--distributed_backend",
        default="ddp",
        type=str,
        help=(
            "Parallelization method for multi-gpu training. Default ddp is strongly advised."
        ),
    )
    parser.add_argument(
        "--no_sequences",
        default=3 * 10 ** 6,
        type=int,
        help="Number of sequences to be used in training and testing. Only used if create_data is true and"
             " tcrb_only is false."
    )

    # gpu/tpu args
    parser.add_argument("--gpus", type=int, default=1, help="How many gpus")
    parser.add_argument("--tpu_cores", type=int, default=None, help="How many tpus")
    parser.add_argument(
        "--val_percent_check",
        default=1.0,
        type=float,
        help=(
            "If you don't want to use the entire dev set (for debugging or "
            "if it's huge), set how much of the dev set you want to use with this flag."
        ),
    )
    parser.add_argument("--only_tcrb", default=False, action="store_true",
                        help="Create and train on only the tcrb files")

    # mixed precision
    parser.add_argument("--precision", type=int, default="32", help="full precision or mixed precision mode")
    parser.add_argument("--amp_level", type=str, default="O1", help="mixed precision type")
    parser.add_argument("--tune_epitope_specificity", default=False, action="store_true")
    parser.add_argument("--folds", default=False, action="store_true")
    parser.add_argument("--embedding_save_name", default="some_emb", type=str)
    parser.add_argument("--add_long_aa", default=-1, type=int)
    parser.add_argument("--relative_data_path", default="./", type=str)
    parser.add_argument("--tune_sp6_labels", default=False, action="store_true")
    parser.add_argument("--train_enc_dec_sp6", default=False, action="store_true")
    parser.add_argument("--train_folds", default=[0, 1], nargs="+")
    parser.add_argument("--run_name", default="generic_run_name", type=str)
    parser.add_argument("--use_glbl_labels", default=False, action="store_true")
    parser.add_argument("--validate_on_mcc", default=False, action="store_true")
    parser.add_argument("--no_pos_enc", default=False, action="store_true")

    # each LightningModule defines arguments relevant to it
    parser = ProtBertClassifier.add_model_specific_args(parser)
    hparams = parser.parse_known_args()[0]
    return hparams, logger


def create_tuning_data(hparams):
    """
        When called without parameter create_data, this method only sets the appropriate file names for the chosen training
        scheme (epitope fine tuning or further bert-like tuning)
    """
    if hparams.tune_epitope_specificity and not hparams.special_tokens:
        print("WARNING!!! Called the training with epitope classifier fine tuning but did not set "
              "the special_tokens parameter. Setting it to true atuomatically...")
        hparams.special_tokens = True
    if hparams.tune_sp6_labels or hparams.train_enc_dec_sp6:
        agnostic_lbls_identifier = "sublbls_" if hparams.use_glbl_labels else ""
        if len(hparams.train_folds) == 3:
            hparams.test_csv, hparams.train_csv, hparams.dev_csv = \
                "sp6_fine_tuning_test_"+ agnostic_lbls_identifier + "{}_{}_{}.csv".format(*hparams.train_folds), \
                "sp6_fine_tuning_train_" + agnostic_lbls_identifier +"{}_{}_{}.csv".format(*hparams.train_folds), \
                "sp6_fine_tuning_valid_"+ agnostic_lbls_identifier +"{}_{}_{}.csv".format(*hparams.train_folds)
        else:
            hparams.test_csv, hparams.train_csv, hparams.dev_csv = \
                "sp6_fine_tuning_test_"+ agnostic_lbls_identifier +"{}_{}.csv".format(hparams.train_folds[0], hparams.train_folds[1]), \
                "sp6_fine_tuning_train_"+ agnostic_lbls_identifier +"{}_{}.csv".format(hparams.train_folds[0], hparams.train_folds[1]), \
                "sp6_fine_tuning_valid_"+ agnostic_lbls_identifier +"{}_{}.csv".format(hparams.train_folds[0], hparams.train_folds[1])

    if hparams.create_data:
        if hparams.tune_epitope_specificity:
            hparams.test_csv, hparams.train_csv, hparams.dev_csv = "epitope_seq_test.csv", "epitope_seq_train.csv", "epitope_seq_valid.csv"
            create_epitope_tuning_files(hparams.relative_data_path)
        elif hparams.tune_sp6_labels:
            create_sp6_tuning_dataset(hparams.relative_data_path, hparams.train_folds)
        elif hparams.train_enc_dec_sp6:
            create_sp6_training_ds(hparams.relative_data_path, hparams.train_folds, hparams.use_glbl_labels)
        else:
            create_bert_further_tuning_files(hparams.relative_data_path)

    if hparams.gradient_checkpointing and hparams.distributed_backend == "ddp":
        print("!!!ERROR!!! When using ddp as the distributed backend method, gradient checkpointing "
              "does not work. Exiting...")
        exit(1)


if __name__ == "__main__":

    hparams, logger = parse_arguments_and_retrieve_logger(save_dir="experiments")
    create_tuning_data(hparams)
    print(hparams.train_csv, hparams.dev_csv, hparams.test_csv)

    logging.getLogger('some_logger')
    logging.basicConfig(filename=hparams.run_name + ".log", level=logging.INFO)
    logging.info("Started training")

    if hparams.gradient_checkpointing and hparams.distributed_backend == "ddp":
        # gradient checkpoint does not work with ddp, which is necessary for multi-gpu training
        print("!!!ERROR!!! When using ddp as the distributed backend method, gradient checkpointing "
              "does not work. Exiting...")
        exit(1)
    model = ProtBertClassifier(hparams)

    if hparams.nr_frozen_epochs == 0:
        model.freeze_encoder()
        model.unfreeze_encoder()

    early_stop_callback = EarlyStopping(
        monitor=hparams.monitor,
        min_delta=0.0,
        patience=hparams.patience,
        verbose=True,
        mode=hparams.metric_mode,
    )

    ckpt_path = os.path.join(logger.save_dir,
        logger.name,
        hparams.run_name,
        "checkpoints",)
    checkpoint_callback = ModelCheckpoint(
        filepath=ckpt_path + "/" + "{epoch}-{val_loss:.2f}-{val_acc:.2f}",
        save_top_k=hparams.save_top_k,
        verbose=True,
        monitor=hparams.monitor,
        period=1,
        mode=hparams.metric_mode,
    )

    trainer = Trainer(
        gpus=hparams.gpus,
        tpu_cores=hparams.tpu_cores,
        logger=logger,
        early_stop_callback=early_stop_callback,
        distributed_backend=hparams.distributed_backend,
        max_epochs=hparams.max_epochs,
        min_epochs=hparams.min_epochs,
        accumulate_grad_batches=hparams.accumulate_grad_batches,
        val_percent_check=hparams.val_percent_check,
        checkpoint_callback=checkpoint_callback,
        precision=hparams.precision,
        amp_level=hparams.amp_level,
        deterministic=True
    )
    trainer.fit(model)
