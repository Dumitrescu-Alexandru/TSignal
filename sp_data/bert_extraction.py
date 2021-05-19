import datetime
# from transformers import AutoTokenizer, AutoModel, pipeline
import pickle
import random
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, RandomSampler

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
    with torch.no_grad():
        acc = sum(y_hat == torch.tensor(labels, device=y_hat.device)).to(dtype=torch.float) / y_hat.shape[0]
    return acc


def remove_ep_lbled_tcrs(data, data_file=None):
    if os.path.exists("/home/dumitra1"):
        folder = "/scratch/work/dumitra1/"
        is_triton = True
    elif os.path.exists("/home/alex"):
        folder = "/home/alex/Desktop/work/covid/covid_tcr_protein_embeddings/data/"
    elif os.path.exists("/u/75/dumitra1"):
        folder = "/u/75/dumitra1/unix/Desktop/aalto_work/covid_tcr_protein_embeddings/data"
    else:
        print("Unrecognized machine. Please add your path files and then continue")
    vdj_data = pd.read_csv(folder + "vdj_human_unique_longs.csv")
    long_vdj_tcrs = set(vdj_data["long"])
    drop_inds = []
    for ind, long_tcr in enumerate(data['Long']):
        if long_tcr in long_vdj_tcrs:
            drop_inds.append(ind)
    data = data.drop(drop_inds)
    print("dropped {} elements from {} dataset".format(len(drop_inds), data_file))
    return data


def read_single_file(file_path, test_inds=[]):
    split_sequences = []
    sep = "\t" if "emerson" in file_path else ","
    data = pd.read_csv(file_path, sep=sep)
    if "emerson" not in file_path:
        data = remove_ep_lbled_tcrs(data)
    sequences = data["Long"]
    for ind, s in enumerate(sequences):
        if ind not in test_inds:
            split_sequences.append(" ".join([s_ for s_ in s]))
    return split_sequences, len(sequences)


def create_epitope_tuning_files():
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

    is_triton = False
    if os.path.exists("/home/dumitra1"):
        folder = "/scratch/work/dumitra1/"
        is_triton = True
    elif os.path.exists("/home/alex"):
        folder = "/home/alex/Desktop/work/covid/data"
    elif os.path.exists("/u/75/dumitra1"):
        folder = "/u/75/dumitra1/unix/Desktop/aalto_work/covid_tcr_protein_embeddings/data"
    else:
        print("Unrecognized machine. Please add your path files and then continue")
        exit(1)
    epitope_dataset = pd.read_csv('../data/vdj_human_unique_longs.csv')
    epitope, sequences = epitope_dataset['epitope'].values, epitope_dataset['long'].values
    all_train_inds, all_val_inds, all_test_inds = split_data(epitope)
    train_seq, train_ep, test_seq, test_ep = sequences[all_train_inds], epitope[all_train_inds], \
                                             sequences[all_test_inds], epitope[all_test_inds]

    train_df = pd.DataFrame(train_seq, train_ep)
    test_df = pd.DataFrame(test_seq, test_ep)
    valid_df = pd.DataFrame(test_seq, test_ep)
    # lblvocab, ind = {}, 0
    # for ep in train_ep:
    #     if ep not in lblvocab:
    #         lblvocab[ep] = ind
    #         ind += 1
    if is_triton:
        train_df.to_csv("/scratch/work/dumitra1/epitope_seq_train.csv")
        test_df.to_csv("/scratch/work/dumitra1/epitope_seq_test.csv")
        valid_df.to_csv("/scratch/work/dumitra1/epitope_seq_valid.csv")
        # pickle.dump(lblvocab, open("/scratch/work/dumitra1/lbl2vocab.bin", "wb"))
    else:
        train_df.to_csv("data/epitope_seq_train.csv")
        test_df.to_csv("data/epitope_seq_test.csv")
        valid_df.to_csv("data/epitope_seq_valid.csv")
        # pickle.dump(lblvocab, open("data/lbl2vocab.bin", "wb"))


def create_bert_files(no_sequences, tcrb_only=False):
    is_triton = False
    if os.path.exists("/home/dumitra1"):
        folder = "/scratch/work/dumitra1/"
        is_triton = True
    elif os.path.exists("/home/alex"):
        folder = "/home/alex/Desktop/work/covid/covid_tcr_protein_embeddings/data/"
    elif os.path.exists("/u/75/dumitra1"):
        folder = "/u/75/dumitra1/unix/Desktop/aalto_work/covid_tcr_protein_embeddings/data"
    else:
        print("Unrecognized machine. Please add your path files and then continue")
        exit(1)

    # adding tcrb_files
    all_data, current_data, elapsed_seqs, no_ds = [], [], 0, 2
    test_inds_ = pickle.load(open("../dataset2_test_inds.bin", "rb"))
    test_inds = {}
    for k, v in test_inds_.items():
        test_inds[k.split("/")[-1]] = v
    seqs, data_len = read_single_file(folder + "/tcrb_human.csv", test_inds['tcrb_human.csv'])
    all_data.extend(seqs)
    elapsed_seqs += data_len
    seqs, data_len = read_single_file(folder + "/tcrb_human_YLQPRTFLL.csv", test_inds['tcrb_human_YLQPRTFLL.csv'])
    all_data.extend(seqs)
    elapsed_seqs += data_len
    for df in os.listdir(folder + "/emerson_long"):
        if not tcrb_only:
            # if only tcrb files are to be used, move ahead and and write datasets with only those
            if df in test_inds:
                seqs, data_len = read_single_file(folder + "/emerson_long/" + df, test_inds[df])
            else:
                seqs, data_len = read_single_file(folder + "/emerson_long/" + df)
            all_data.extend(seqs)
            elapsed_seqs += data_len
            no_ds += 1
        if elapsed_seqs > no_sequences or tcrb_only:
            test_valid_inds = random.sample(list(range(len(all_data))), int(len(all_data) * 0.2))
            test_inds = test_valid_inds[:len(test_valid_inds) // 2]
            valid_inds = test_valid_inds[len(test_valid_inds) // 2:]
            test_valid_inds = set(test_valid_inds)
            test_data = [all_data[i] for i in test_inds]
            valid_data = [all_data[i] for i in valid_inds]
            all_data = [all_data[i] for i in range(len(all_data)) if i not in test_valid_inds]
            all_data = pd.DataFrame(all_data)
            valid_data = pd.DataFrame(valid_data)
            test_data = pd.DataFrame(test_data)
            if is_triton:
                all_data.to_csv("/scratch/work/dumitra1/tcr_seqs_train_df.csv")
                valid_data.to_csv("/scratch/work/dumitra1/tcr_seqs_dev_df.csv")
                test_data.to_csv("/scratch/work/dumitra1/tcr_seqs_test_df.csv")
            else:
                all_data.to_csv("data/tcr_seqs_train_df.csv")
                valid_data.to_csv("data/tcr_seqs_dev_df.csv")
                test_data.to_csv("data/tcr_seqs_test_df.csv")

            return


class EpitopeClassifierDataset(Dataset):

    def __init__(self, path) -> None:
        self.data = []
        if os.path.exists("/home/dumitra1"):
            path = "/scratch/work/dumitra1/" + path
            vocab = pickle.load(open("/scratch/work/dumitra1/lbl2vocab.bin", "rb"))
        else:
            path = "data/" + path
            vocab = pickle.load(open("data/lbl2vocab.bin", "rb"))
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


class BertDataset(Dataset):
    """
    Loads the Dataset from the csv files passed to the parser.
    :param hparams: HyperOptArgumentParser obj containg the path to the data files.
    :param train: flag to return the train set.
    :param val: flag to return the validation set.
    :param test: flag to return the test set.
    Returns:
        - Training Dataset, Development Dataset, Testing Dataset
    """

    def __init__(self, path, special_tokens) -> None:
        self.data = []
        if os.path.exists("/home/dumitra1"):
            path = "/scratch/work/dumitra1/" + path
        else:
            path = "data/" + path
        self.init_dataset(path, special_tokens)

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

        self.modelFolderPath = 'models/ProtBert/'
        self.vocabFilePath = os.path.join(self.modelFolderPath, 'vocab.txt')

        self.extract_emb = False
        self.metric_acc = gpu_acc_metric
        # self.special_tokens = hparams.special_tokens
        # self.tune_epitope_specificity = hparams.tune_epitope_specificity

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
        self.encoder_features = 1024

        # Tokenizer
        self.tokenizer = BertTokenizer(self.vocabFilePath, do_lower_case=False)

        # Label Encoder
        self.label_encoder = LabelEncoder(
            self.hparams.label_set.split(","), reserved_labels=[]
        )
        self.label_encoder.unknown_index = None

        # Classification head
        # self.classification_head = nn.Sequential(
        #     nn.Linear(self.encoder_features, self.label_encoder.vocab_size),
        #     nn.Tanh(),
        # )
        # Classification for further bert training
        if self.hparams.tune_epitope_specificity:
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
    def get_epitope_weights():
        vdj_long_data = pd.read_csv("../data/vdj_human_unique_longs.csv")
        epitope2ind = pickle.load(open("data/lbl2vocab.bin", "rb"))
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
        if self.hparams.tune_epitope_specificity:
            weights = self.get_epitope_weights()
            self._loss = nn.CrossEntropyLoss(weight=torch.tensor(weights))
        else:
            self._loss = nn.CrossEntropyLoss()

    def unfreeze_encoder(self) -> None:
        """ un-freezes the encoder layer. """
        if self._frozen:
            log.info(f"\n-- Encoder model fine-tuning")
            # for param in self.ProtBertBFD.parameters():
            #     if np.random.rand() > 0.8:
            #         param.requires_grad = True
            for name, param in self.ProtBertBFD.named_parameters():
                # if "encoder" in name or "pooler" in name:
                #     param.requires_grad = True
                # else:
                #     param.requires_grad = False
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

    def forward(self, input_ids, token_type_ids, attention_mask, target_positions=None, return_embeddings=False):
        """ Usual pytorch forward function.
        :param tokens: text sequences [batch_size x src_seq_len]
        :param lengths: source lengths [batch_size]
        Returns:
            Dictionary with model outputs (e.g: logits)
        """
        input_ids = torch.tensor(input_ids, device=self.device)
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

    def encode_labels(self, labels):
        if self.hparams.tune_epitope_specificity:
            return labels
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
        # sample = sample[0]
        # seq, t, tp = sample["seq"], sample["target"], sample["target_pos"]
        # tp = set(tp)
        # # print(seq, t, tp)
        # for ind, s in enumerate(seq.split(" ")):
        #     if s == "[MASK]":
        #         if ind + 1 not in tp:
        #             print("WARNIGN")
        #         else:
        #             print("Nah is cul. Look at collate_tensor function")
        sample = collate_tensors(sample)
        inputs = self.tokenizer.batch_encode_plus(sample["seq"],
                                                  add_special_tokens=self.hparams.special_tokens,
                                                  padding=True,
                                                  truncation=True,
                                                  max_length=self.hparams.max_length)

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

        tqdm_dict = {"val_loss": val_loss_mean, "val_acc": val_acc_mean}
        result = {
            "progress_bar": tqdm_dict,
            "log": tqdm_dict,
            "val_loss": val_loss_mean,
        }
        return result

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
        optimizer = optim.Adam(parameters, lr=self.hparams.learning_rate)
        return [optimizer], []

    def on_epoch_end(self):
        """ Pytorch lightning hook """
        if self.current_epoch + 1 >= self.nr_frozen_epochs:
            self.unfreeze_encoder()

    def __retrieve_dataset(self, train=True, val=True, test=True):
        """ Retrieves task specific dataset """
        if self.hparams.tune_epitope_specificity:
            if train:
                return EpitopeClassifierDataset(hparams.train_csv)
            elif val:
                return EpitopeClassifierDataset(hparams.dev_csv)
            elif test:
                return EpitopeClassifierDataset(hparams.test_csv)
            else:
                print('Incorrect dataset split')
        else:
            if train:
                return BertDataset(hparams.train_csv, hparams.special_tokens)
            elif val:
                return BertDataset(hparams.dev_csv, hparams.special_tokens)
            elif test:
                return BertDataset(hparams.test_csv, hparams.special_tokens)
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
            default=5e-6,
            type=float,
            help="Encoder specific learning rate.",
        )
        parser.add_argument(
            "--learning_rate",
            default=3e-5,
            type=float,
            help="Classification head learning rate.",
        )
        parser.opt_list(
            "--nr_frozen_epochs",
            default=1,
            type=int,
            help="Number of epochs we want to keep the encoder model frozen.",
            tunable=True,
            options=[0, 1, 2, 3, 4, 5],
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


def setup_testube_logger() -> TestTubeLogger:
    """ Function that sets the TestTubeLogger to be used. """
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y--%H-%M-%S")
    if os.path.exists("/home/dumitra1"):
        save_dir = "/scratch/work/dumitra1/experiments/"
    else:
        save_dir = "experiments/"
    return TestTubeLogger(
        save_dir=save_dir,
        version=dt_string,
        name="lightning_logs",
    )


logger = setup_testube_logger()

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
parser.add_argument("--only_tcrb", default=False, action="store_true", help="Create and train on only the tcrb files")

# mixed precision
parser.add_argument("--precision", type=int, default="32", help="full precision or mixed precision mode")
parser.add_argument("--amp_level", type=str, default="O1", help="mixed precision type")
parser.add_argument("--tune_epitope_specificity", default=False, action="store_true")
parser.add_argument("--embedding_save_name",default="some_emb", type=str)
parser.add_argument("--add_long_aa",default=-1, type=int)

# each LightningModule defines arguments relevant to it
parser = ProtBertClassifier.add_model_specific_args(parser)
hparams = parser.parse_known_args()[0]
if hparams.tune_epitope_specificity and not hparams.special_tokens:
    print("WARNING!!! Called the training with epitope classifier fine tuning but did not set "
          "the special_tokens parameter. Setting it to true atuomatically...")
    hparams.special_tokens = True
if hparams.create_data:
    print(hparams.test_csv)
    if hparams.tune_epitope_specificity:
        hparams.test_csv, hparams.train_csv, hparams.dev_csv = "epitope_seq_test.csv", "epitope_seq_train.csv", "epitope_seq_valid.csv"
        create_epitope_tuning_files()
    else:
        print("Extracting and saving {} milion sequences for training".format(hparams.no_sequences))
        create_bert_files(hparams.no_sequences * 10 ** 6, hparams.only_tcrb)

model = ProtBertClassifier(hparams)
if hparams.nr_frozen_epochs == 0:
    model.freeze_encoder()
    model.unfreeze_encoder()

if hparams.gradient_checkpointing and hparams.distributed_backend == "ddp":
    print("!!!ERROR!!! When using ddp as the distributed backend method, gradient checkpointing "
          "does not work. Exiting...")
    exit(1)
# ------------------------
# 2 INIT EARLY STOPPING
# ------------------------
early_stop_callback = EarlyStopping(
    monitor=hparams.monitor,
    min_delta=0.0,
    patience=hparams.patience,
    verbose=True,
    mode=hparams.metric_mode,
)

# --------------------------------
# 3 INIT MODEL CHECKPOINT CALLBACK
# -------------------------------
ckpt_path = os.path.join(
    logger.save_dir,
    logger.name,
    f"version_{logger.version}",
    "checkpoints",
)
# initialize Model Checkpoint Saver
checkpoint_callback = ModelCheckpoint(
    filepath=ckpt_path + "/" + "{epoch}-{val_loss:.2f}-{val_acc:.2f}",
    save_top_k=hparams.save_top_k,
    verbose=True,
    monitor=hparams.monitor,
    period=1,
    mode=hparams.metric_mode,
)

# ------------------------
# 4 INIT TRAINER
# ------------------------
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


def preprocess_tcr_data(tcrs0, epis0, minutcrs=80, useall=False, long0=None):
    """Select only unique TCRs for each epitope. Use only epitopes that have at least minutcrs unique TCRs"""
    all_categories0 = np.unique(epis0)
    tcrs, num_tcrs4epis, epis_u, labels, long = [], [], [], [], []

    for epi in all_categories0:
        I = epis0 == epi
        num = sum(I)
        if useall:
            long += list(long0[I])
            tcrs += list(tcrs0[I])
            epis_u.append(epi)
            num_tcrs4epis.append(num)
        elif num > minutcrs:
            tis = list(np.unique(tcrs0[I]))
            longtis = list(np.unique(long0[I]))
            num = len(tis)
            if num > minutcrs:
                tcrs += tis
                long += longtis
                epis_u.append(epi)
                num_tcrs4epis.append(num)
    n_categories = len(epis_u)

    # print(num_tcrs4epis)
    icat = 0
    istart = 0
    for i, tcr in enumerate(tcrs):
        if i >= istart + num_tcrs4epis[icat]:
            istart += num_tcrs4epis[icat]
            icat += 1

        labels_i = np.zeros((1, n_categories), dtype=int)
        labels_i[0, icat] = 1

        #        print(i,istart)
        for j, tj in enumerate(tcrs[:istart]):
            if tcr == tj:
                labels_i = np.bitwise_or(labels_i, labels[j])
                labels[j] = labels_i
        labels.append(labels_i)
    tcrs = np.asarray(tcrs)
    labels = np.concatenate(labels, axis=0)

    return tcrs, labels, epis_u, num_tcrs4epis, long


def load_data_new_data():
    if os.path.exists("vdj_human_unique50.csv"):
        filename = "vdj_human_unique50.csv"
    elif os.path.exists("../vdj_human_unique50.csv"):
        filename = "../vdj_human_unique50.csv"
    elif os.path.exists("data/vdj_human_unique50.csv"):
        filename = "data/vdj_human_unique50.csv"
    elif os.path.exists("../data/vdj_human_unique50.csv"):
        filename = "../data/vdj_human_unique50.csv"
    else:
        filename = None
        exit(1)
    data = pd.read_csv(filename)
    tcr_data, epitope, long_data = data['CDR3B'], data['Epitope'], data['Long']
    return tcr_data, epitope, long_data


def load_data():
    # VDJDB DATA, 22 EPITOPES, CONFIDENCE SCORE >=1
    filename = '../data/vdj_human_unique_longs.csv'
    epis, subject, cdr3s, long = np.loadtxt(filename, usecols=(0, 1, 5, 7), unpack=True, delimiter=',', skiprows=1,
                                            comments=None,
                                            dtype='str')
    # Remove the 3 TCRs that were removed from the later version of VDJdb
    # 143 CASSLFVGGPGNEQFF ['NLVPMVATV']
    # 146 CASSPPAGSYNEQFF ['NLVPMVATV']
    # 155 CASTGTSGALYNEQFF ['NLVPMVATV']
    I = np.ones((len(cdr3s),), dtype=bool)
    I[[143, 146, 155]] = 0
    epis, cdr3s, long = epis[I], cdr3s[I], long[I]
    tcrs, labels_ar, epis_u, num_tcrs4epis, long = preprocess_tcr_data(cdr3s, epis, minutcrs=0, useall=True,
                                                                       long0=long)
    n_categories = labels_ar.shape[1]
    return tcrs, labels_ar, epis_u, num_tcrs4epis, n_categories, subject, long


def read_noncontrol_data(data_file="../data/tcrb_human.csv", dictionary=None):
    data = pd.read_csv(data_file)
    tcr_data, subjects, epitope, long_data = data['CDR3B'].values, data['Subject'].values, \
                                             data['Epitope'].values, data['Long'].values
    non_control_indices = []
    for i, s in enumerate(subjects):
        if s != "control":
            non_control_indices.append(i)
    tcr_data, epitope, long_data = tcr_data[non_control_indices], epitope[non_control_indices], \
                                   long_data[non_control_indices]
    return tcr_data, epitope, long_data


def extract_and_save_embeddings_for_clustering(model, return_first=False, emb_name="some_mdl", batch_size=100):
    model.cuda()
    try:
        data = pd.read_csv("/scratch/work/dumitra1/tcrb_human.csv")
    except:
        data = pd.read_csv("/scratch/work/dumitra1/tcrb_human.csv")
    tcr_data, subjects, epitope, long_data = data['CDR3B'].values, data['Subject'].values, \
                                             data['Epitope'].values, data['Long'].values
    non_control_indices = []
    for i, s in enumerate(subjects):
        if s != "control":
            non_control_indices.append(i)
    tcr_data, epitope, long_data = tcr_data[non_control_indices], epitope[non_control_indices], \
                                   long_data[non_control_indices]
    ld2ep = {}
    for ep, ld in zip(epitope, long_data):
        ld2ep[ld] = ep
    sequences = []
    for t, l in zip(tcr_data, long_data):
        sequences.append(' '.join(l[i] for i in range(len(l))))
    inputs = model.tokenizer.batch_encode_plus(sequences,
                                               add_special_tokens=True,
                                               padding=True,
                                               truncation=True,
                                               max_length=model.hparams.max_length)
    vdjdb_embeddings = {}
    count = 0
    model.extract_emb = True
    model.eval()
    for i in range(0, len(sequences) // batch_size + 1):
        current_inputs = {}
        seqs = sequences[i * batch_size:(i + 1) * batch_size]
        current_inputs['input_ids'] = inputs['input_ids'][i * batch_size:(i + 1) * batch_size]
        current_inputs['token_type_ids'] = inputs['token_type_ids'][i * batch_size:(i + 1) * batch_size]
        current_inputs['attention_mask'] = inputs['attention_mask'][i * batch_size:(i + 1) * batch_size]
        embedding = model.forward(**current_inputs)
        embedding = np.array(embedding)
        print("{} sequences have been extracted".format(i * batch_size))
        features = []
        if return_first:
            return embedding[-1], seqs[-1]
        for seq_num in range(len(embedding)):
            seq_len = len(seqs[seq_num].replace(" ", ""))
            start_Idx = 1
            end_Idx = seq_len + 1
            seq_emd = embedding[seq_num][start_Idx:end_Idx]
            features.append(seq_emd)
        for ind in range(len(seqs)):
            long_seq = seqs[ind].replace(" ", "")
            if long_seq not in ld2ep:
                count += 1
            else:
                start_cdr3, end_cdr3 = long_data[i * batch_size + ind].find(tcr_data[i * batch_size + ind]), \
                                       long_data[i * batch_size + ind].find(tcr_data[i * batch_size + ind]) + \
                                       len(tcr_data[i * batch_size + ind])
                clean_feat = features[start_cdr3:end_cdr3]
                vdjdb_embeddings[tcr_data[ind]] = (clean_feat, ld2ep[long_seq])

    print("Count", count)
    pickle.dump(vdjdb_embeddings, open("/scratch/work/dumitra1/emb_name" + ".bin", "wb"))


def extract_perp_score(model):
    model.to(torch.device("cuda:0"))
    model.eval()

    def generate_masked_seqs(seq):
        all_seqs = []
        lbl, lbl_pos = [], []
        for i in range(len(seq)):
            lbl.append(seq[i])
            lbl_pos.append([i + 1])
            all_seqs.append(' '.join(seq[j] if j != i else "[MASK]" for j in range(len(seq))))
        return all_seqs, lbl, lbl_pos

    seqs = pd.read_csv('test_perplexity_data.csv').values
    batch_size = hparams.batch_size
    sum_neg_probs, N = 0, 0
    for i in range(len(seqs) // batch_size):
        if i * batch_size == 100:
            with open("log_perp_test.txt", "at") as f:
                f.write("{} sequences have elapse".format(i * batch_size))
        current_seqs = seqs[i * batch_size: (i + 1) * batch_size]
        sequences, all_lbls, all_lbl_pos = [], [], []
        for s in current_seqs:
            seqs_, lbl, lbl_pos = generate_masked_seqs(s[-1])
            sequences.extend(seqs_)
            all_lbls.extend(lbl)
            all_lbl_pos.extend(lbl_pos)
        all_lbls = model.encode_labels(all_lbls)
        inputs = model.tokenizer.batch_encode_plus(sequences,
                                                   add_special_tokens=hparams.special_tokens,
                                                   padding=True,
                                                   truncation=True,
                                                   max_length=model.hparams.max_length)
        out = model(**inputs, target_positions=all_lbl_pos)
        out = torch.nn.functional.softmax(out)
        out = out.detach().cpu().numpy()
        target_probs = out[list(range(out.shape[0])), all_lbls[0]]
        neg_log_probs = -np.log(np.array(target_probs))
        sum_neg_probs += np.sum(neg_log_probs.reshape(-1))
        N += len(all_lbls[0])
        if i == 2:
            break
    final_perp_score = np.exp((1 / N) * sum_neg_probs)

    with open("log_perp_test.txt", "at") as f:
        f.write("Perplexity score is  {} with {} sequences".format(final_perp_score, N))


def load_mira_data(file="../data/ImmuneCODE_MIRA/immuneCODE_MIRA_TCRs.tsv"):
    data = pd.read_csv(file, sep="\t")
    tcrs, epitope = data['CRD3B'], data['Epitope groups']
    return list(tcrs), list(epitope)


def remove_duplicates(long, cdr3, epitope):
    seen_cdr3s = set()
    long_u, cdr3_u, epitope_u = [], [], []
    for l, c, e in zip(long, cdr3, epitope):
        if c not in seen_cdr3s:
            seen_cdr3s.add(c)
            long_u.append(l)
            cdr3_u.append(c)
            epitope_u.append(e)
    return long_u, cdr3_u, epitope_u


def extract_and_save_embeddings(model, return_first=False, emb_name="some_mdl", test_new_data=False,
                                extract_cdr3=False, use_only_cdr3=False, use_covid_data="",
                                add_specific_epitopes_vdj50=[], sum_tcrs=True, extract_unlabeled=True,
                                add_long_aa=-1, data_file="raw_seq_data_0.15_0.9.bin"):
    """
    Function for various bert-model based embedding extraction
    sum_tcrs: sums tcrs along the sequence dimension, to reduce memory necessary (only useful probably in visualizations
            with umap/t-SNE
    extract_unlabeled: retrieves seqs from tcrb_human and emerson that do not have an epitope specificity and extracts
            the embeddings for those
    use_covid_data: new covid files
    use_only_cdr3: do not use the whole long sequence when extracting the cdr3b sequences (useful when for some datapoints
                theres only the CDR3B available etc.)
    """
    model.cuda()
    if test_new_data:
        long, epitope = pickle.load(open(data_file, "rb"))
        cdr3bs = long
        # when extracting covid data, those only hve CDR3B seqs.
    elif use_covid_data:
        file = "../data/ImmuneCODE_MIRA/immuneCODE_MIRA_ci_TCRs.tsv" if use_covid_data == "ci" \
            else "../data/ImmuneCODE_MIRA/immuneCODE_MIRA_cii_TCRs.tsv"
        cdr3bs, epitope = load_mira_data(file)
        long = cdr3bs
        cdr3bs, epitope, long = list(cdr3bs), list(epitope), list(long)
        long, cdr3, epitope = remove_duplicates(long, cdr3bs, epitope)
    else:
        cdr3bs, labels_ar, epis_u, num_tcrs4epis, n_categories, subject, long = load_data()
        cdr3bs = list(cdr3bs)
        long = list(long)
        ind_label = np.argmax(labels_ar, 1)
        epitope = [epis_u[ind_label[i]] for i in range(len(ind_label))]
    if add_specific_epitopes_vdj50:
        print(len(cdr3bs), len(long))
        vdj50_data = pd.read_csv("../data/vdj_human_unique50.csv")
        vdj_50_cdr3bs, vdj_50_epitopes, vdj_50_longs = vdj50_data['CDR3B'].values, vdj50_data['Epitope'].values, vdj50_data['Long'].values
        for ind, (t, e, l) in enumerate(zip(vdj_50_cdr3bs, vdj_50_epitopes, vdj_50_longs)):
            if e in add_specific_epitopes_vdj50:
                epitope.append(e)
                cdr3bs.append(t)
                long.append(l)
    if extract_unlabeled:
        tcrb_data = pd.read_csv("../data/tcrb_human.csv")
        none_inds_tcrb = np.argwhere(tcrb_data['Epitope'].values == 'none').reshape(-1)
        none_tcrs_tcrb, none_long_tcrbs = tcrb_data['CDR3B'].values[none_inds_tcrb], \
                                          tcrb_data['Long'].values[none_inds_tcrb]
        samples = random.sample(list(range(len(none_tcrs_tcrb))), 20000)
        none_tcrs_tcrb, none_long_tcrbs = none_tcrs_tcrb[samples], none_long_tcrbs[samples]
        none_epitopes_tcrb = ['none'] * len(none_tcrs_tcrb)

        emerson_data = pd.read_csv("../data/emerson_long/emerson_long_1.tsv", sep='\t')
        samples = random.sample(list(range(len(emerson_data))), 20000)
        none_tcrs_emerson, none_long_emerson = emerson_data['CDR3B'].values, emerson_data['Long'].values
        none_tcrs_emerson, none_long_emerson = none_tcrs_emerson[samples], none_long_emerson[samples]
        none_epitopes_tcrb = ['none'] * len(none_tcrs_tcrb)
        none_epitopes_emerson = ['none'] * len(none_long_emerson)
        cdr3bs.extend(none_tcrs_tcrb)
        cdr3bs.extend(none_tcrs_emerson)
        long.extend(none_long_tcrbs)
        long.extend(none_long_emerson)
        epitope.extend(none_epitopes_tcrb)
        epitope.extend(none_epitopes_emerson)

    ld2ep, l2cdr3 = {}, {}
    u_e = set()
    for ep, ld in zip(epitope, long):
        ld2ep[ld] = ep
        u_e.add(ep)
    for ld, cdr3 in zip(long, cdr3bs):
        l2cdr3[ld] = cdr3
    cropped_seqs, sequences = [], []
    for t, l in zip(cdr3bs, long):
        if add_long_aa != -1:
            cdr_start, cdr_end = l.find(t), l.find(t) + len(t)
            cropeed_seq = l[cdr_start - add_long_aa: cdr_end + add_long_aa]
            cropped_seqs.append(' '.join(cropeed_seq[i] for i in range(len(cropeed_seq))))
        else:
            cropped_seqs.append(' '.join(l[i] for i in range(len(l))))
        sequences.append(l)
    inputs = model.tokenizer.batch_encode_plus(cropped_seqs,
                                               add_special_tokens=hparams.special_tokens,
                                               padding=True,
                                               truncation=True,
                                               max_length=model.hparams.max_length)
    print(len(long), len(epitope), len(cdr3bs))

    vdjdb_embeddings = {}
    count = 0
    model.extract_emb = True
    model.eval()
    file_index = 0
    for i in range(0, len(sequences) // 100 + 1):
        current_inputs = {}
        seqs = sequences[i * 100:(i + 1) * 100]
        crped_seqs = cropped_seqs[i * 100:(i+1) * 100]
        current_inputs['input_ids'] = inputs['input_ids'][i * 100:(i + 1) * 100]
        current_inputs['token_type_ids'] = inputs['token_type_ids'][i * 100:(i + 1) * 100]
        current_inputs['attention_mask'] = inputs['attention_mask'][i * 100:(i + 1) * 100]
        embedding = model.forward(**current_inputs)
        embedding = embedding.cpu().numpy()
        print("{} sequences have been extracted".format(i * 100))
        features = []
        if return_first:
            return embedding[-1], seqs[-1]
        for seq_num in range(len(embedding)):
            seq_len = len(seqs[seq_num].replace(" ", ""))
            start_Idx = 1 if hparams.special_tokens else 0
            end_Idx = seq_len + 1 if hparams.special_tokens else seq_len
            seq_emd = embedding[seq_num][start_Idx:end_Idx]
            if extract_cdr3 and not use_only_cdr3:
                ld = seqs[seq_num]
                ld_cropped = crped_seqs[seq_num].replace(" ", "")
                cdr3 = l2cdr3[ld]
                start_cdr3, end_cdr3 = ld_cropped.find(cdr3), ld_cropped.find(cdr3) + len(cdr3)
                seq_emd = seq_emd[start_cdr3:end_cdr3]
            if sum_tcrs:
                features.append(np.sum(seq_emd, axis=0, keepdims=True))
            else:
                features.append(seq_emd)
        for ind in range(len(seqs)):
            long_seq = seqs[ind]
            if long_seq not in ld2ep:
                vdjdb_embeddings[long_seq] = (features[ind], "ASD")
            else:
                vdjdb_embeddings[long_seq] = (features[ind], ld2ep[long_seq])
        if i * 100 % 3000 == 0 and i != 0:
            pickle.dump(vdjdb_embeddings, open(emb_name + "_{}.bin".format(file_index), "wb"))
            file_index += 1
            vdjdb_embeddings = {}
    if vdjdb_embeddings:
        pickle.dump(vdjdb_embeddings, open(emb_name + "_{}.bin".format(file_index), "wb"))
    # print("Count", count)
    #
    #     vdjdb_embeddings = {}
    # for ind in range(len(sequences)):
    #     vdjdb_embeddings[sequences[ind].replace(" ","")] = features[ind]
    # pickle.dump(vdjdb_embeddings, open(emb_name + ".bin", "wb"))


# trainer.fit(model)
# best_checkpoint_path = "experiments/lightning_logs/version_08-02-2021--14-24-31/" \
#                        "checkpoints/epoch=19-val_loss=2.73-val_acc=0.33.ckpt"
# model = model.load_from_checkpoint(best_checkpoint_path)
# extract_perp_score(model)

# model.eval()
# model.freeze()
# extract_and_save_embeddings(model, emb_name="MIRA_ci_original_bert_cdr3Only", test_new_data=False, extract_cdr3=True,
#                             use_only_cdr3=True, use_covid_data="ci")
# extract_and_save_embeddings(model, emb_name="MIRA_cii_original_bert_cdr3Only", test_new_data=False, extract_cdr3=True,
#                             add_specific_epitopes_vdj50=['NEGVKAAW', 'LLQTGIHVRVSQPSL', 'YSEHPTFTSQY', 'AMFWSVPTV'])
if os.path.exists("/scratch/work"):
    hparams.embedding_save_name = "/scratch/work/dumitra1/" + hparams.embedding_save_name
extract_and_save_embeddings(model, emb_name="bert_seq_data_0.9_0.15", test_new_data=True, extract_cdr3=True,
                            sum_tcrs=False, extract_unlabeled=False,add_long_aa=-1, data_file="raw_seq_data_0.9_0.15.bin")

# extract_and_save_embeddings(model, emb_name="vdj50_original_bert_cdr3Only", test_new_data=True, extract_cdr3=True,
#                             use_only_cdr3=True, use_covid_data="")
#
# extract_and_save_embeddings_for_clustering(model, emb_name="original_bert")
