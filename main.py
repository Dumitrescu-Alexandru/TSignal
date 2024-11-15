import pickle
import os
import datetime
from sp_data.data_utils import SPbinaryData
from train_scripts.cv_train_cs_predictors import train_cs_predictors, test_seqs_w_pretrained_mdl, train_sp_type_predictor,\
    test_w_precomputed_sptypes
import argparse
import logging

def create_param_set_cs_predictors():

    from sklearn.model_selection import ParameterGrid

    parameters = {"train_folds": [[0,1],[0,2],[1,2]], "nlayers": [3],"nheads":[16],
                  "lr": [0.00001], 'dropout':[0.1], 'random_folds':[True], 'train_on_subset':[0.25,0.5,0.75,1],
                  "run_number":[3, 4, 5]}
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
    parser.add_argument("--lr", default=0.0001, type=float, help="Learning rate for TSignal (not ProtBERT).")
    parser.add_argument("--anneal_start", default=-1, type=int, help="Epoch where learning rate starts to be lowered.")
    parser.add_argument("--anneal_epochs", default=-1, type=int, help="Number of epochs in which to reach the desired annealed_lr")
    parser.add_argument("--annealed_lr", default=0.00002, type=float, help="Desired annealed_lr")
    parser.add_argument("--param_set_search_number", default=-1, type=int, help="Create set of grid-search parameters "
                "with create_param_set_cs_predictors and index them. Save the index2parameters in a binary and on a specific"
                "run, load the param set specified by this argument (useful with e.g. slurm parallel runs)")
    parser.add_argument("--train_cs_predictor", default=False, action="store_true", help="Train a CS predictor like TSignal"
                         " (alternatively, can also train a binary SP type classifier)")
    parser.add_argument("--batch_size", default=32, type=int, help="Training batch size")
    parser.add_argument("--run_name", default="some_run", type=str, help="Name your run. This will be used as a prefix for all saved files: logs, models, etc.")
    parser.add_argument("--epochs", default=-1, type=int, help="By default, model uses tolerence based stopping criteria. "
                           "Set this for a fixed number of epochs training.")
    parser.add_argument("--add_og_info", default=False, action="store_true", help="Use organism group information for the"
                                                                                  "trained models.")
    parser.add_argument("--dropout", default=0, type=float, help="Dropout on all TSignal layers (not ProtBERT).")
    parser.add_argument("--use_glbl_lbls", default=False, action="store_true", help="Train a separate head/completely "
                            "separate model taking ProtBERT's E_30 embeddings and predict the SP type. TSignal will"
                            "then used that prediction (will set y1=pred of glbl_lbl_predictor) and predict the rest of seq from there.")
    parser.add_argument("--nlayers", default=3, type=int, help="Number of layers for TSignal (3 layer on encoder, 3 for"
                   "         decoder when using enc-dec transformer, or 3-layered decoder when using param train_only_decoder is true)")
    parser.add_argument("--nheads", default=8, type=int, help="Number of heads to be used (same as paranthesis for nlayers)")
    parser.add_argument("--patience", default=30, type=int, help="``Static'' patience (patience is not resetted on finding a better run. "
                             "When patience number of epochs yield worse validation, stop and retrieve the best checkpoint)")
    parser.add_argument("--train_oh", default=False, action="store_true", help="Use one-hot encoding instead of ProtBERT")
    parser.add_argument("--train_folds", default=[0,1], nargs="+", help="Folds to train on. Need to have 3 separate runs trained"
                                "on [0,1], [0,2], [1,2] if the paper experiments are to be repeated")
    parser.add_argument("--deployment_model", default=False, action="store_true", help="If training a final (deployment) model,"
                                   "the training fold will be [0,1,2] and the model is only validated (not tested at then end)")
    parser.add_argument("--test_seqs", default="", type=str, help="filename of the binary containing sequences that will be tested. File needs to be in sp_data/ folder")
    parser.add_argument("--test_mdl", default="", type=str, help="filename of model that will be tested on test_seqs file")
    parser.add_argument("--lr_scheduler_swa", default="none", type=str, help="when using swa, one option is to cycle the learning rate after n training steps (instead of after each epoch)."
                                     "")
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
    parser.add_argument("--frozen_pe_epochs", default=-1, type=int)
    parser.add_argument("--no_bert_pe_training", default=False, action="store_true")
    parser.add_argument("--extended_sublbls", default=False, action="store_true")
    parser.add_argument("--random_folds", default=False, action="store_true")
    parser.add_argument("--train_on_subset", default=1., type=float)
    parser.add_argument("--train_only_decoder", default=False, action="store_true")
    parser.add_argument("--remove_bert_layers", default=0, type=int)
    parser.add_argument("--augment_trimmed_seqs", default=False, action="store_true")
    parser.add_argument("--saliency_map_save_fn", default="save.bin", type=str)
    parser.add_argument("--compute_saliency", default=False, action="store_true")
    parser.add_argument("--hook_layer", default="bert", type=str)
    parser.add_argument("--cycle_length", default=5, type=int)
    parser.add_argument("--bert_pe_for_decoder", default=False, action="store_true")
    parser.add_argument("--concat_pos_enc", default=False, action="store_true")
    parser.add_argument("--pe_extra_dims", default=64, type=int)
    parser.add_argument("--add_bert_pe_from_dec_to_bert_out", default=False, action="store_true")
    parser.add_argument("--lr_multiplier_swa", default=1, type=int, help="number to multiply the maximum swa lr")
    parser.add_argument("--change_swa_decoder_optimizer", default=False, action="store_true", help="change decoder optimizer when starting swa from Adam to SGD")
    parser.add_argument("--reinint_swa_decoder", default=False, action="store_true", help="Use the same (Adam) optimizer when SWA starts, but reinitialize it")
    parser.add_argument("--add_val_data_on_swa", default=False, action="store_true", help="SWA isnt based on early stopping, so the validation data can also be used for training")
    parser.add_argument("--lipbobox_predictions", default=False, action="store_true", help="Modify predictions to have SSS...(SSS)LLL for Sec/SPase (I)II and TTT...(TTT)LLL for Tat/SPase (I)II")
    parser.add_argument("--train_sp_type_predictor", default=False, action="store_true", help="train a TCR conv architecture to predict the sp type")
    parser.add_argument("--load_model", default="none", type=str, help="Option to load any bert model (e.g. trained on SP-CS prediction task)")
    parser.add_argument("--deep_mdl", default=False, action="store_true", help="Add another layer before the class head of the sp type model")
    parser.add_argument("--is_cnn2", default=False, action="store_true", help="Remove the additional linear layer.")
    parser.add_argument("--no_of_layers_onlysp", default=8, type=int, help="No. of additional layers sp classifier.")
    parser.add_argument("--test_sptype_preds", default="none", type=str, help="File name for sp type predictor dictionary results."
                                                                              "Use predictions from binary sp type classifier as first label preds for a model.")
    parser.add_argument("--no_of_layers_conv_resnets", default=4, type=int, help="No. of additional (conv) layers sp classifier.")
    parser.add_argument("--is_cnn4", default=False, action="store_true", help="Use new cnn4 architecture")
    parser.add_argument("--use_sgd_on_swa", default=False, action="store_true", help="Use sgd on swa for sp type classifier")
    parser.add_argument("--swa_start", default=60,type=int, help="Epoch at wich you load best mdl and start swa")
    parser.add_argument("--og_emb_dim", default=32,type=int, help="Epoch at wich you load best mdl and start swa")
    parser.add_argument("--residue_emb_extra_dims", default=0,type=int, help="Epoch at wich you load best mdl and start swa")
    parser.add_argument("--add_extra_embs2_decoder", default=False,action="store_true", help="Add the extra decoder inputs to the Transformer Decoder (this way, the final layer W_O still won't have access to non-contextualized residue representations")
    parser.add_argument("--use_blosum", default=False,action="store_true")
    parser.add_argument("--use_extra_oh", default=False,action="store_true")
    parser.add_argument("--use_aa_len", default=200, type=int, help="When training binary classifiers (not for TSignal).")
    parser.add_argument("--data", default="mammal", type=str, help="When training binary classifiers (not for TSignal).")
    parser.add_argument("--output_file", default="", type=str, help="Specify output file name")
    parser.add_argument("--verbouse", default=False, action="store_true", help="If true, print results when using test_seqs_w_pretrained_mdl")

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
    if 'lr_scheduler_swa' in param_set:
        args.lr_sceduler = param_set['lr_scheduler_swa']
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

if __name__ == "__main__":
    date_now = str(datetime.datetime.now()).split(".")[0].replace("-", "").replace(":", "").replace(" ", "")
    logging.getLogger('some_logger')
    args = parse_arguments()
    if args.test_mdl and args.test_sptype_preds != "none":
        test_w_precomputed_sptypes(args)
    if args.test_seqs:
        test_seqs_w_pretrained_mdl(args.test_mdl, args.test_seqs, tune_bert=args.tune_bert, saliency_map_save_fn=args.saliency_map_save_fn,hook_layer=args.hook_layer, compute_saliency=args.compute_saliency,
                                   output_file=args.output_file, verbouse=args.verbouse)
    elif args.train_sp_type_predictor:
        logging.basicConfig(filename=args.run_name + ".log", level=logging.INFO)

        args2 = parse_arguments()
        train_sp_type_predictor(args2)
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
        a = train_cs_predictors(args)
