import numpy as np
import random
import pickle
import torch.nn as nn
import torch.optim as optim
import torch

from sp_data.data_utils import SPbinaryData, BinarySPDataset
from models.binary_sp_classifier import BinarySPClassifier


def weighted_avg_results_for_dataset(test_results_for_ds):
    num_batches_per_ds = [td[0] for td in test_results_for_ds]
    weights = [nb / sum(num_batches_per_ds) for nb in num_batches_per_ds]
    negative_result = sum([weights[i] * test_results_for_ds[i][1] for i in range(len(weights))])
    positive_result = sum([weights[i] * test_results_for_ds[i][2] for i in range(len(weights))])
    return negative_result, positive_result


def get_pos_neg_for_datasets(test_datasets, model, data_folder, use_aa_len=200, epoch=0):
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
            if use_aa_len != 200:
                x = x[:, :use_aa_len, :]
            with torch.no_grad():
                model_preds = model(x.permute(0, 2, 1))
            positive_preds, negative_preds = model_preds >= 0.5, model_preds < 0.5
            positive_pred_inds, negative_pred_inds = torch.nonzero(positive_preds).reshape(-1).detach().cpu().numpy(), \
                                                     torch.nonzero(negative_preds).reshape(-1).detach().cpu().numpy()
            actual_positives, actual_negatives = y >= 0.5, y < 0.5
            actual_positives_inds, actual_negatives_inds = torch.nonzero(actual_positives).reshape(
                -1).detach().cpu().numpy(), \
                                                           torch.nonzero(actual_negatives).reshape(
                                                               -1).detach().cpu().numpy()
            positive_hits = len(set(positive_pred_inds).intersection(actual_positives_inds))
            negative_hits = len(set(negative_pred_inds).intersection(actual_negatives_inds))
            total_positive_hits += positive_hits
            total_negative_hits += negative_hits
            total_positives += len(actual_positives_inds)
            total_negatives += len(actual_negatives_inds)
        test_results_for_ds.append((len(dataset_loader), total_negative_hits / total_negatives,
                                    total_positive_hits / total_positives, epoch))
    neg, pos = weighted_avg_results_for_dataset(test_results_for_ds)
    return neg, pos, test_results_for_ds


def train_fold(train_datasets, test_datasets, data_folder, model, param_set, fixed_ep_test=-1):
    """

    :param train_datasets: list of train ds file names
    :param test_datasets: list of test ds file names
    :param data_folder: folder where embedding datasets are found
    :param model: initialized model object
    :param param_set: dictionary of parameters
    :param fixed_ep_test: if != -1, test after this many epochs (used in nested-cv, when number of epochs to train is
                          tuned on the training set
    :return: dictionary with maximum results TODO return the model and test it in nested-cv case
    """
    lr, patience, use_aa_len = param_set['lr'], param_set['patience'], param_set['use_aa_len']
    optimizer = optim.Adam(lr=lr, params=model.parameters())
    criterion = nn.BCELoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    convergence_condition = False
    max_results, epoch, max_pos, max_neg, epochs_trained = {}, -1, 0, 0, 0
    while not convergence_condition:
        epoch += 1
        model.train()
        for train_ds in train_datasets:
            dataset = BinarySPDataset(data_folder + train_ds)
            dataset_loader = torch.utils.data.DataLoader(dataset,
                                                         batch_size=64, shuffle=True,
                                                         num_workers=4)
            for ind, batch in enumerate(dataset_loader):
                x, y = batch['emb'], batch['lbl']
                x, y = x.to(device), y.to(device).to(torch.float)
                if use_aa_len != 200:
                    x = x[:, :use_aa_len, :]
                preds = model(x.permute(0, 2, 1))
                optimizer.zero_grad()
                loss = criterion(preds.reshape(-1), y)
                loss.backward()
                optimizer.step()
        if fixed_ep_test != -1 and fixed_ep_test == epoch:
            neg, pos, test_results_for_ds = get_pos_neg_for_datasets(test_datasets, model, data_folder, use_aa_len,
                                                                     epoch)
        else:
            neg, pos, test_results_for_ds = get_pos_neg_for_datasets(test_datasets, model, data_folder, use_aa_len,
                                                                     epoch)
            if pos < max_pos:
                patience -= 1
            else:
                max_neg, max_pos, max_results = neg, pos, test_results_for_ds
        print("Results for epoch {} (pos/neg acc): {}/{}".format(epoch, pos, neg))
        convergence_condition = patience == 0

    return max_results


def init_model(parameters=None):
    model = BinarySPClassifier(input_size=1024, output_size=1)
    return model


def train_test_folds(run_name, train_datasets_per_fold, test_datasets_per_fold, data_folder, param_set, nested=False,
                     fixed_epoch=-1):
    """
    :param run_name: save name
    :param train_datasets_per_fold: list of training dataset filenames
    :param test_datasets_per_fold: list of testing dataset filenames
    :param data_folder: path to where the datasets are found
    :param param_set: the current parameter set
    :param nested: if nested, the model is not train-testing on the outer cross-val set. The testing is also done on a
                    separate test set, used to select the best model. The final results will then be saved from the
                    outer test fold-loop
    :param fixed_epoch: when != -1, the model training does not use patience, but rather train of this ammount of epochs.
                        used in nested-cv: after the best set of params is found in the inner-cv loop, along with the
                        hyperparameters, the number of epochs it was trained for is also returned (that is also tuned with
                        patience) and that is the fixed_epoch parameter
    :return: list containing (number_of_datapoints, negative_acc, pos_acc, epoch)
    """
    results_over_all_ds = []
    for ind, (train_datasets, test_datasets) in enumerate(zip(train_datasets_per_fold,
                                                              test_datasets_per_fold)):
        model = init_model()
        max_results = train_fold(train_datasets, test_datasets, data_folder, model, param_set)
        if not nested:
            pickle.dump([train_datasets, test_datasets, max_results], open("{}_results_on_fold_{}.bin".
                                                                           format(run_name, ind), "wb"))
        results_over_all_ds.extend(max_results)
    return results_over_all_ds


def split_train_test(train_datasets):
    """
    Function used to further split the training datasets into 4-folds of 75% training, 25% test data  for nested-cv
    :param train_datasets:
    :return:
    """
    test_ds_number = int(0.25 * len(train_datasets))
    remaining_ds = len(train_datasets) - test_ds_number * 4
    number_of_test_ds_per_fold = []
    train_ds_subfold, test_ds_subfold = [], []
    for i in range(4):
        if remaining_ds:
            number_of_test_ds_per_fold.append(test_ds_number + 1)
            remaining_ds -= 1
        else:
            number_of_test_ds_per_fold.append(test_ds_number)
    remaining_untested_datasets = set(train_datasets)
    ind = 0
    while remaining_untested_datasets:
        current_test_ds_subfold = random.sample(remaining_untested_datasets, number_of_test_ds_per_fold[ind])
        current_train_ds_subfold = list(set(train_datasets) - set(current_test_ds_subfold))
        ind += 1
        remaining_untested_datasets = remaining_untested_datasets - set(current_test_ds_subfold)
        test_ds_subfold.append(current_test_ds_subfold)
        train_ds_subfold.append(current_train_ds_subfold)
    return train_ds_subfold, test_ds_subfold


def get_avg_results_for_fold(results):
    total_ds_counts = sum([results[i][0] for i in range(len(results))])
    weights = [results[i][0] / total_ds_counts for i in range(len(results))]
    negative_results = sum([results[i][1] * weights[i] for i in range(len(results))])
    positive_results = sum([results[i][2] * weights[i] for i in range(len(results))])
    avg_epoch = int(np.mean([results[i][3] for i in range(len(results))]))
    return negative_results, positive_results, avg_epoch


def train_test_nested_folds(run_name, params, train_ds, test_ds, data_folder):
    best_param, best_result = None, 0
    # move over all 80%/20% train/test splits
    best_pos, best_results_params_and_epoch = 0, []
    for ind, (train_datasets, test_datasets) in enumerate(zip(train_ds,
                                                              test_ds)):
        # for param_set in params:
        #     # further split into 4 folds of 75%/25% the training set
        #     print(train_datasets)
        #     train_ds_subfold, test_ds_subfold = split_train_test(train_datasets)
        #     print(train_datasets)
        #     max_results = train_test_folds(run_name, train_ds_subfold, test_ds_subfold, data_folder, param_set,
        #                                    nested=True)
        #     print(max_results)
        #     current_neg, current_pos, train_epochs = get_avg_results_for_fold(max_results)
        #     if best_pos < current_pos:
        #         best_pos = current_pos
        #         best_results_params_and_epoch = [param_set, train_epochs]
        # final_result_current_fold = train_test_folds(run_name, train_datasets, test_datasets,
        #                                              param_set=best_results_params_and_epoch[0],
        #                                              fixed_epoch=best_results_params_and_epoch[1],
        #                                              data_folder=data_folder)
        parameters = {'lr': 0.0001, 'patience': 1, 'use_aa_len': 100}
        model = init_model(parameters)
        final_result_current_fold = train_fold( train_datasets, test_datasets,data_folder=data_folder,model=model,
                                                     param_set=parameters,
                                                     fixed_ep_test=0)
        final_result_current_fold = get_avg_results_for_fold(final_result_current_fold)
        print(final_result_current_fold)
        pickle.dump([final_result_current_fold, best_results_params_and_epoch], open("results_fold_{}.bin".format(ind), "rb"))



def train_bin_sp_mdl(run_name, use_aa_len, lr, nested_cv=True, parameters=None):
    sp_data = SPbinaryData()
    train_ds, test_ds, data_folder = sp_data.train_datasets_per_fold, \
                                     sp_data.test_datasets_per_fold, sp_data.data_folder
    if parameters is None:
        parameters = [{'lr': 0.0001, 'patience': 1, 'use_aa_len': 100}]

    if nested_cv:
        train_test_nested_folds(run_name, parameters, train_ds, test_ds, data_folder)
    else:
        train_test_folds(run_name, train_ds, test_ds, data_folder, parameters[0])
# for nested cv
# >>> for i in range(len(a) // tes_ds_n + 1):
# ...     print(a[i*tes_ds_n : (i+1)*tes_ds_n])
