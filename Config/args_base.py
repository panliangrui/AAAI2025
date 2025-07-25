import argparse
from Config.base import *

def get_argp(dataname,model_name):

    argp = argparse.ArgumentParser()
    argp.add_argument("--model_name", default=model_name, type=str)
    argp.add_argument("--dataname", default=dataname,type=str)
    argp.add_argument("--train_data",default=f"/root/autodl-tmp/STAS/{dataname}/features_gigapath/",type=str)
    argp.add_argument("--test_data",default=f"/root/autodl-tmp/STAS/{dataname}/features_gigapath/",type=str)
    argp.add_argument("--val_data",default=f"/root/autodl-tmp/STAS/{dataname}/features_gigapath/",type=str)

    argp.add_argument("--val",default=True,type=bool)

    argp.add_argument("--nclass", default=1, type=int)
    argp.add_argument("--input_dim", default=1536, type=int)
    argp.add_argument("--hidden_dim", default=512, type=int)

    argp.add_argument("--train_bs", default=1)
    argp.add_argument("--val_bs",default=1)
    argp.add_argument("--test_bs", default=1)

    argp.add_argument("--num_workers", default=0)

    argp.add_argument("--lr", default=1e-4)
    argp.add_argument("--wd", default=1e-5)

    argp.add_argument("--start_epoch", default=0, type=int)
    argp.add_argument("--num_epochs", default=50, type=int)
    argp.add_argument("--start_seed",default=0,type=int)
    argp.add_argument("--end_seed",default=5,type=int)
    argp.add_argument("--epoch_frq",default=-1,type=str)
    argp.add_argument("--patient", default=100)


    argp.add_argument("--tensorboard_dir", default="./{}/log".format(dataname))
    argp.add_argument("--checkpoint_dir", default="./{}/checkpoint".format(dataname))
    argp.add_argument("--metic_dir", default="./{}/".format(dataname))
    argp.add_argument("--metric",default="auc")
    argp.add_argument("--seed", default=0)
    argp.add_argument("--device", default="cuda")
    args = argp.parse_args()
    return args