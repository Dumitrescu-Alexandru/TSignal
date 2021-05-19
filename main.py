import datetime
from sp_data.data_utils import SPbinaryData
from train_scripts.cv_train_binary_sp_model import train_bin_sp_mdl
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_aa_len", default=200, type=int)
    parser.add_argument("--run_name", default="run", type=str)
    parser.add_argument("--lr", default=0.001, type=float)
    return parser.parse_args()

if __name__=="__main__":
    date_now = str(datetime.datetime.now()).split(".")[0].replace("-", "").replace(":", "").replace(" ", "")
    args = parse_arguments()
    run_name = args.run_name + "_" + date_now
    a = train_bin_sp_mdl(args.run_name,args.use_aa_len,args.lr)