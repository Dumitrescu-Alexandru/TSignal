import pickle
import os
import datetime
from sp_data.data_utils import SPbinaryData
from train_scripts.cv_train_binary_sp_model import train_bin_sp_mdl
import argparse


def create_parameter_set():
    from sklearn.model_selection import ParameterGrid
    parameters = {"dos": [[0., 0.], [0.1, 0.1], [0.2, 0.2], [0.3, 0.3]], "filters":[[100, 80, 60, 40],
                    [120, 100, 80, 60], [140, 120, 100, 80]], "lengths":[[3, 5, 9, 15],[5, 9, 15, 21]],
                  "lr":[0.001, 0.0001], "patience":[10], "use_aa_len":[100]}
    group_params = list(ParameterGrid(parameters))
    grpid_2_params = {}
    for i in range(len(group_params) // 10  + 1):
        grpid_2_params[i] = group_params[i*10:(i+1)*10]
    pickle.dump(grpid_2_params, open("param_groups_by_id.bin", "wb"))

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_aa_len", default=200, type=int)
    parser.add_argument("--run_name", default="run", type=str)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--param_set_search_number", default=-1, type=int)
    return parser.parse_args()


if __name__ == "__main__":
    date_now = str(datetime.datetime.now()).split(".")[0].replace("-", "").replace(":", "").replace(" ", "")
    args = parse_arguments()
    if args.param_set_search_number != -1 and not os.path.exists("param_groups_by_id.bin"):
        create_parameter_set()
    run_name = args.run_name + "_" + date_now
    a = train_bin_sp_mdl(args.run_name, args.use_aa_len, args.lr)
