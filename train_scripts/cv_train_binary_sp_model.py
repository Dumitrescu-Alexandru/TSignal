import pickle
import torch.nn as nn
import torch.optim as optim
import torch

from sp_data.data_utils import SPbinaryData, BinarySPDataset
from models.binary_sp_classifier import BinarySPClassifier

def weighted_avg_results_for_dataset(test_results_for_ds):
    num_batches_per_ds = [td[0] for td in test_results_for_ds]
    weights = [nb/sum(num_batches_per_ds) for nb in num_batches_per_ds]
    negative_result = sum([weights[i] * test_results_for_ds[i][1] for i in range(len(weights))])
    positive_result = sum([weights[i] * test_results_for_ds[i][2] for i in range(len(weights))])
    return negative_result, positive_result

def get_pos_neg_for_datasets(test_datasets, model, data_folder):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_results_for_ds = []
    for test_ds in test_datasets:
        model.eval()
        dataset = BinarySPDataset(data_folder + test_ds)
        dataset_loader = torch.utils.data.DataLoader(dataset,
                                                     batch_size=64, shuffle=True,
                                                     num_workers=4)
        total_positives, total_negatives = 0, 0
        total_positive_hits, total_negative_hits = 0, 0
        for batch in dataset_loader:
            x, y = batch['emb'], batch['lbl']
            x, y = x.to(device), y.to(device).to(torch.float)
            with torch.no_grad():
                model_preds = model(x.permute(0,2,1))
            positive_preds, negative_preds = model_preds >= 0.5, model_preds  < 0.5
            positive_pred_inds, negative_pred_inds = torch.nonzero(positive_preds).reshape(-1).detach().cpu().numpy(),\
                                                     torch.nonzero(negative_preds).reshape(-1).detach().cpu().numpy()
            actual_positives, actual_negatives = y >= 0.5, y < 0.5
            actual_positives_inds, actual_negatives_inds = torch.nonzero(actual_positives).reshape(-1).detach().cpu().numpy(),\
                                                           torch.nonzero(actual_negatives).reshape(-1).detach().cpu().numpy()
            positive_hits = len( set(positive_pred_inds).intersection(actual_positives_inds) )
            negative_hits = len( set(negative_pred_inds).intersection(actual_negatives_inds) )
            total_positive_hits += positive_hits
            total_negative_hits += negative_hits
            total_positives += len(actual_positives_inds)
            total_negatives += len(actual_negatives_inds)
        test_results_for_ds.append((len(dataset_loader),total_negative_hits/total_negatives,
                                        total_positive_hits/total_positives))
    neg, pos = weighted_avg_results_for_dataset(test_results_for_ds)
    return neg, pos, test_results_for_ds

def train_fold(train_datasets, test_datasets, data_folder, model, lr=0.0001, patience=5):
    optimizer = optim.Adam(lr=lr, params=model.parameters())
    criterion = nn.BCELoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    convergence_condition = False
    max_dict, epoch, max_pos, max_neg = {}, 0, 0, 0
    while not convergence_condition:
        model.train()
        for train_ds in train_datasets:
            dataset = BinarySPDataset(data_folder + train_ds)
            dataset_loader = torch.utils.data.DataLoader(dataset,
                                                             batch_size=64, shuffle=True,
                                                             num_workers=4)
            for ind, batch in enumerate(dataset_loader):
                x, y = batch['emb'], batch['lbl']
                x, y = x.to(device), y.to(device).to(torch.float)
                preds = model(x.permute(0,2,1))
                optimizer.zero_grad()
                loss = criterion(preds.reshape(-1), y)
                # if ind % 20 == 0:
                #     print("Loss on batch {}: {}".format(ind, loss.item()))
                loss.backward()
                optimizer.step()
        neg, pos, test_results_for_ds = get_pos_neg_for_datasets(test_datasets, model, data_folder)
        if pos < max_pos:
            patience -= 1
        else:
            max_neg, max_pos, max_dict = neg, pos, test_results_for_ds
        print("Results for epoch {} (pos/neg acc): {}/{}".format(epoch, pos, neg))
        convergence_condition = patience == 0

    return max_dict

def init_model():
    model = BinarySPClassifier(input_size=1024, output_size=1)
    return model

def train_bin_sp_mdl():
    sp_data = SPbinaryData()
    model = init_model()
    for ind, (train_datasets, test_datasets) in enumerate(zip(sp_data.train_datasets_per_fold,
                                                            sp_data.test_datasets_per_fold)):
        max_dict = train_fold(train_datasets, test_datasets, sp_data.data_folder, model)
        pickle.dump([train_datasets, test_datasets, max_dict], open("results_on_fold_{}.bin", "wb"))



