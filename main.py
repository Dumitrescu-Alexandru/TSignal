import pickle
import os
import datetime
from sp_data.data_utils import SPbinaryData
from train_scripts.cv_train_cs_predictors import train_cs_predictors, test_seqs_w_pretrained_mdl
from train_scripts.cv_train_binary_sp_model import train_bin_sp_mdl
import argparse
import logging

def create_param_set_cs_predictors():

    from sklearn.model_selection import ParameterGrid
    # parameters = { "lr_scheduler":["step", "expo"], "train_folds":[[0,1],[1,2],[0,2]],fparam_set_search_number
    #               "run_number":list(range(5))}
    # parameters = {"lr_sched_warmup":[0, 10], "lr_scheduler":["step", "expo"], "train_folds":[[0,1],[1,2],[0,2]],
    #               "run_number":list(range(5))}
    # parameters = {'run_number':list(range(5)), "train_folds":[[0,1],[1,2],[0,2]],
    #               'run_name':["glbl_lbl_search_"], 'use_glbl_lbls':[1], 'glbl_lbl_weight':[0.1, 1],
    #               'glbl_lbl_version':[1,2]}
    # parameters = {"wd":[0., 0.0001, 0.00001], "train_folds":[[0,1],[1,2],[0,2]] }

    parameters = {"train_folds": [[0,1],[0,2],[1,2]], "nlayers": [3],"nheads":[16],
                  "lr": [0.00001], 'dropout':[0.1], 'random_folds':[True], 'train_on_subset':[0.25,0.5,0.75,1],
                  "run_number":[3, 4, 5]}
    # parameters = {"dos":[0.],"nlayers": [4], "ff_d": [4096], "nheads":[4],
    #               "lr": [0.00001], "train_folds":[[0,1],[0,2],[1,2]], "run_number":list(range(10))}
    group_params = list(ParameterGrid(parameters))
    grpid_2_params = {}
    for i in range(len(group_params)):
        grpid_2_params[i] = group_params[i]
    # start_id = len(group_params)

    # add 5 without-glbl label runs. See if glbl label actually helps the TAT/LIPO metrics
    pickle.dump(grpid_2_params, open("param_groups_by_id_cs.bin", "wb"))

def create_parameter_set():
    from sklearn.model_selection import ParameterGrid
    parameters = {"dos": [[0.1, 0.2], [0.3, 0.3]], "filters":[[100, 80, 60, 40],
                    [120, 100, 80, 60], [140, 120, 100, 80]], "lengths":[[3, 5, 9, 15],[5, 9, 15, 21]],
                  "lr":[0.001, 0.0001], "patience":[5], "use_aa_len":[100], "pos_weight":[1,5]}
    group_params = list(ParameterGrid(parameters))
    grpid_2_params = {}
    for i in range(len(group_params) // 5  + 1):
        grpid_2_params[i] = group_params[i*5:(i+1)*5]
    pickle.dump(grpid_2_params, open("param_groups_by_id_cs.bin", "wb"))

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_aa_len", default=200, type=int)
    parser.add_argument("--lr", default=0.00001, type=float)
    parser.add_argument("--high_lr", default=False, action="store_true", help="increase 10x the lr until swa stars")
    parser.add_argument("--data", default="mammal", type=str)
    parser.add_argument("--param_set_search_number", default=-1, type=int)
    parser.add_argument("--train_cs_predictor", default=False, action="store_true")
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--run_name", default="some_run", type=str)
    parser.add_argument("--epochs", default=-1, type=int, help="By default, model uses tolerence. Set this for a fixed"
                                                               "number of epochs training")
    parser.add_argument("--add_lg_info", default=False, action="store_true")
    parser.add_argument("--dropout", default=0, type=float)
    parser.add_argument("--test_freq", default=5, type=int)
    parser.add_argument("--use_glbl_lbls", default=False, action="store_true")
    parser.add_argument("--nlayers", default=3, type=int)
    parser.add_argument("--nheads", default=8, type=int)
    parser.add_argument("--patience", default=30, type=int)
    parser.add_argument("--train_oh", default=False, action="store_true")
    parser.add_argument("--train_folds", default=[0,1], nargs="+")
    parser.add_argument("--deployment_model", default=False, action="store_true")
    parser.add_argument("--test_seqs", default="", type=str)
    parser.add_argument("--test_mdl", default="", type=str)
    parser.add_argument("--lr_scheduler", default="none", type=str)
    parser.add_argument("--lr_sched_warmup", default=0, type=int)
    parser.add_argument("--ff_d", default=4096, type=int, help='Expanding dimension')
    parser.add_argument("--test_beam", default=False, action="store_true")
    parser.add_argument("--wd", default=0., type=float)
    parser.add_argument("--glbl_lbl_weight", default=1., type=float)
    parser.add_argument("--glbl_lbl_version", default=1, type=int)
    parser.add_argument("--validate_on_test", default=False, action="store_true")
    parser.add_argument("--form_sp_reg_data", default=False, action="store_true")
    parser.add_argument("--simplified", default=True, action="store_true")
    parser.add_argument("--very_simplified", default=True, action="store_true")
    parser.add_argument("--version2_agregation", default="max",type=str)
    parser.add_argument("--validate_partition", default=None,type=int)
    parser.add_argument("--validate_on_mcc", default=False,action="store_true")
    parser.add_argument("--tune_cs", default=0, type=int)
    parser.add_argument("--input_drop", default=False, action="store_true")
    parser.add_argument("--use_swa", default=False, action="store_true")
    parser.add_argument("--separate_save_sptype_preds", default=False, action="store_true")
    parser.add_argument("--no_pos_enc", default=False, action="store_true")
    parser.add_argument("--linear_pos_enc", default=False, action="store_true")
    parser.add_argument("--scale_input", default=False, action="store_true")
    parser.add_argument("--test_only_cs", default=False, action="store_true")
    parser.add_argument("--weight_class_loss", default=False, action="store_true")
    parser.add_argument("--weight_lbl_loss", default=False, action="store_true")
    parser.add_argument("--account_lipos", default=False, action="store_true")
    parser.add_argument("--tuned_bert_embs", default=False, action="store_true")
    parser.add_argument("--tune_bert", default=False, action="store_true", help="Tune BERT and TSignal together")
    parser.add_argument("--frozen_epochs", default=3, type=int)
    parser.add_argument("--extended_sublbls", default=False, action="store_true")
    parser.add_argument("--random_folds", default=False, action="store_true")
    parser.add_argument("--train_on_subset", default=1., type=float)
    parser.add_argument("--train_only_decoder", default=False, action="store_true")
    parser.add_argument("--remove_bert_layers", default=0, type=int)
    parser.add_argument("--augment_trimmed_seqs", default=False, action="store_true")
    parser.add_argument("--saliency_map_save_fn", default="save.bin", type=str)
    parser.add_argument("--hook_layer", default="bert", type=str)
    parser.add_argument("--cycle_length", default=5, type=int)
    parser.add_argument("--lr_multiplier_swa", default=1, type=int, help="number to multiply the maximum swa lr")
    parser.add_argument("--change_swa_decoder_optimizer", default=False, action="store_true", help="change decoder optimizer when starting swa from Adam to SGD")
    parser.add_argument("--reinint_swa_decoder", default=False, action="store_true", help="Use the same (Adam) optimizer when SWA starts, but reinitialize it")
    parser.add_argument("--add_val_data_on_swa", default=False, action="store_true", help="SWA isnt based on early stopping, so the validation data can also be used for training")

    return parser.parse_args()

def modify_param_search_args(args):
    params = pickle.load(open("param_groups_by_id_cs.bin", "rb"))
    param_set = params[args.param_set_search_number]
    run_name = args.run_name
    if 'run_name' in param_set:
        run_name = param_set['run_name']
    run_name += "_"
    if 'patience' in param_set:
        run_name += "patience_{}".format(param_set['patience'])
        args.patience = param_set['patience']
    if "test_beam" in param_set:
        args.test_beam = param_set["test_beam"]
    if 'use_glbl_lbls' in param_set:
        args.use_glbl_lbls = param_set['use_glbl_lbls']
    if args.use_glbl_lbls:
        run_name += "use_glbl_lbls_"
        version = param_set['glbl_lbl_version'] if 'glbl_lbl_version' in param_set else args.glbl_lbl_version
        if param_set['use_glbl_lbls']:
            run_name += "use_glbl_lbls_version_{}_".format(version)
        args.glbl_lbl_version = version
        if 'glbl_lbl_weight' in param_set:
            args.glbl_lbl_weight = param_set['glbl_lbl_weight']
            run_name += "weight_{}_".format(param_set['glbl_lbl_weight'])
    if 'dos' in param_set:
        args.dropout = param_set['dos']
        run_name += "dos_{}_".format(args.dropout)
    if "dropout" in param_set:
        args.dropout = param_set['dropout']
        run_name += "dos_{}_".format(args.dropout)
    if 'lr' in param_set:
        args.lr = param_set['lr']
        run_name += "lr_{}_".format(args.lr)
    if 'ff_d' in param_set:
        args.ff_d = param_set['ff_d']
        run_name += "ffd_{}_".format(args.ff_d)
    if 'nlayers' in param_set:
        args.nlayers = param_set['nlayers']
        run_name += "nlayers_{}_".format(args.nlayers)
    if 'nheads' in param_set:
        args.nhead = param_set['nheads']
        run_name += "nhead_{}_".format(args.nhead)
    if 'wd' in param_set:
        args.wd = param_set['wd']
        run_name += "wd_{}_".format(args.wd)
    if 'lr_scheduler' in param_set:
        args.lr_sceduler = param_set['lr_scheduler']
        run_name += "lrsched_{}_".format(args.lr_sceduler)
    if 'lr_sched_warmup' in param_set:
        args.lr_sched_warmup = param_set['lr_sched_warmup']
        run_name += "wrmpLrSched_{}_".format(param_set['lr_sched_warmup'])
    if "run_number" in param_set:
        run_name += "run_no_{}_".format(param_set['run_number'])
    if 'train_on_subset' in param_set:
        args.train_on_subset = param_set['train_on_subset']
        run_name += "subset_train_{}_".format(param_set['train_on_subset'])
    if 'train_folds' in param_set:
        args.train_folds = param_set['train_folds']
    if 'validate_partition' in param_set:
        args.validate_partition = param_set['validate_partition']

    # use the train folds in the name of the model regardless
    if args.validate_partition is not None:
        run_name += "_t_{}_v_{}".format(args.train_folds[0], args.validate_partition)
    else:
        run_name += "trFlds_{}_{}".format(args.train_folds[0], args.train_folds[1])
    args.run_name = run_name
    return args

def sanity_check(file, args2):
    params = pickle.load(open(file, "rb"))
    param_names = []
    param_logs = []
    param_bins = []
    param_bins_best = []
    for k,v in params.items():
        # if k in range(180,216):
        args2.param_set_search_number = k
        args2 = modify_param_search_args(args2)
        param_names.append(args2.run_name + "_best_eval.pth")
        param_logs.append(args2.run_name + ".log")
        param_bins.append(args2.run_name + ".bin")
        param_bins_best.append(args2.run_name + "_best.bin")
        args2 = parse_arguments()
    if len(param_names) != len(set(param_names)):
        print("WARNING: THE NUMBER OF UNIQUE MODEL NAMES IS NOT EQUAL TO THE NUMBER OF PARAMETERS! EXITING...")
        exit(1)
    # print(" ".join(param_names))
    # print(" ".join(param_logs))
    # print(" ".join(param_bins))
    # print(" ".join(param_bins_best))
    # exit(1)

if __name__ == "__main__":
    date_now = str(datetime.datetime.now()).split(".")[0].replace("-", "").replace(":", "").replace(" ", "")
    logging.getLogger('some_logger')
    args = parse_arguments()
    if args.test_seqs:
        test_seqs_w_pretrained_mdl(args.test_mdl, args.test_seqs, tune_bert=args.tune_bert, saliency_map_save_fn=args.saliency_map_save_fn,hook_layer=args.hook_layer)
    elif args.train_cs_predictor:
        args2 = parse_arguments()
        if not os.path.exists("param_groups_by_id_cs.bin"):
            create_param_set_cs_predictors()
        if args.param_set_search_number != -1:
            sanity_check("param_groups_by_id_cs.bin", args2)
            args = modify_param_search_args(args)
        elif args.validate_partition is not None:
            args.run_name += "_t_{}_v_{}".format(args.train_folds[0], args.validate_partition)
        logging.basicConfig(filename=args.run_name + ".log", level=logging.INFO)
        logging.info("Started training")
        args.train_folds = [int(tf) for tf in args.train_folds]
        a = train_cs_predictors(bs=args.batch_size, eps=args.epochs, run_name=args.run_name, use_lg_info=args.add_lg_info,
                                lr=args.lr, dropout=args.dropout, test_freq=args.test_freq, use_glbl_lbls=args.use_glbl_lbls,
                                ff_d=args.ff_d, partitions=args.train_folds, nlayers=args.nlayers, nheads=args.nheads, patience=args.patience,
                                train_oh=args.train_oh, deployment_model=args.deployment_model, lr_scheduler=args.lr_scheduler,
                                lr_sched_warmup=args.lr_sched_warmup, test_beam=args.test_beam, wd=args.wd,
                                glbl_lbl_weight=args.glbl_lbl_weight,glbl_lbl_version=args.glbl_lbl_version,
                                validate_on_test=args.validate_on_test, form_sp_reg_data=args.form_sp_reg_data,
                                simplified=args.simplified, version2_agregation=args.version2_agregation,
                                validate_partition=args.validate_partition,very_simplified=args.very_simplified,
                                validate_on_mcc=args.validate_on_mcc, tune_cs=args.tune_cs, input_drop=args.input_drop,
                                use_swa=args.use_swa, separate_save_sptype_preds=args.separate_save_sptype_preds,
                                no_pos_enc=args.no_pos_enc, linear_pos_enc=args.linear_pos_enc, scale_input=args.scale_input,
                                test_only_cs=args.test_only_cs, weight_class_loss=args.weight_class_loss, weight_lbl_loss=args.weight_lbl_loss,
                                account_lipos=args.account_lipos, tuned_bert_embs=args.tuned_bert_embs,
                                tune_bert=args.tune_bert, frozen_epochs=args.frozen_epochs, extended_sublbls=args.extended_sublbls,
                                random_folds=args.random_folds, train_on_subset=args.train_on_subset, train_only_decoder=args.train_only_decoder,
                                remove_bert_layers=args.remove_bert_layers, augment_trimmed_seqs=args.augment_trimmed_seqs,
                                high_lr=args.high_lr, cycle_length=args.cycle_length,lr_multiplier_swa=args.lr_multiplier_swa,
                                change_swa_decoder_optimizer=args.change_swa_decoder_optimizer, add_val_data_on_swa=args.add_val_data_on_swa,
                                reinint_swa_decoder=args.reinint_swa_decoder)

    else:
        if args.param_set_search_number != -1 and not os.path.exists("param_groups_by_id.bin"):
            create_parameter_set()
        run_name = args.run_name + "_" + date_now
        a = train_bin_sp_mdl(args.run_name, args.use_aa_len, args.lr, data=args.data, nested_cv=args.data == "mammal",)
