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
    # parameters = { "lr_scheduler":["step", "expo"], "train_folds":[[0,1],[1,2],[0,2]],
    #               "run_number":list(range(5))}
    # parameters = {"lr_sched_warmup":[0, 10], "lr_scheduler":["step", "expo"], "train_folds":[[0,1],[1,2],[0,2]],
    #               "run_number":list(range(5))}
    # parameters = {'run_number':list(range(5)), "train_folds":[[0,1],[1,2],[0,2]],
    #               'run_name':["glbl_lbl_search_"], 'use_glbl_lbls':[1], 'glbl_lbl_weight':[0.1, 1],
    #               'glbl_lbl_version':[1,2]}
    # parameters = {"wd":[0., 0.0001, 0.00001], "train_folds":[[0,1],[1,2],[0,2]] }

    parameters = {"train_folds": [[0,1],[0,2],[1,2]], "nlayers": [3,5,8],"nheads":[8,16],
                  "lr": [0.00001], 'use_glbl_lbls':[1], 'glbl_lbl_version':[2], 'patience':[30],
                  'dropout':[0,0.3,0.5]}
    # parameters = {"dos":[0.],"nlayers": [4], "ff_d": [4096], "nheads":[4],
    #               "lr": [0.00001], "train_folds":[[0,1],[0,2],[1,2]], "run_number":list(range(10))}
    group_params = list(ParameterGrid(parameters))
    # add 5 without-glbl label runs. See if glbl label actually helps the TAT/LIPO metrics
    grpid_2_params = {}
    for i in range(len(group_params)):
        grpid_2_params[i] = group_params[i]
    # start_id = len(group_params)

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
        test_seqs_w_pretrained_mdl(args.test_mdl, args.test_seqs)
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
                                validate_partition=args.validate_partition,very_simplified=args.very_simplified)

    else:
        if args.param_set_search_number != -1 and not os.path.exists("param_groups_by_id.bin"):
            create_parameter_set()
        run_name = args.run_name + "_" + date_now
        a = train_bin_sp_mdl(args.run_name, args.use_aa_len, args.lr, data=args.data, nested_cv=args.data == "mammal",)
