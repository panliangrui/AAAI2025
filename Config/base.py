import os
from club import util



def build_dict(datanames,root,last):
    cur_dict={}
    for name in datanames:
        cur_dict[name]=os.path.join(root,name,last)
    return cur_dict

root="/root/autodl-tmp/plr_sata_aaai/"

def set_data(args):
    util.fix_random_seed(args.seed)

    args.train_excel_path=os.path.join(root,"data",args.dataname,f"fold{args.seed}_train.csv")
    args.val_excel_path=os.path.join(root,"data",args.dataname,f"fold{args.seed}_val.csv")
    args.test_excel_path=os.path.join(root,"data",args.dataname,f"fold{args.seed}_test.csv")

    args.test_best_pt_path="{}/{}_best_seed{}.pth".format(args.checkpoint_dir,args.model_name,args.seed)
    args.test_last_pt_path="{}/{}_last_seed{}.pth".format(args.checkpoint_dir,args.model_name,args.seed)

    args.test_best_csv_path = "{}/test_kflod_best.csv".format(args.metic_dir)
    args.test_last_csv_path = "{}/test_kflod_Last.csv".format(args.metic_dir)


def reset_run_root(args,run_root):
    if not os.path.exists(run_root):
        os.makedirs(run_root)

    args.tensorboard_dir = "{}/{}/{}/log".format(run_root,args.model_name, args.dataname)
    args.checkpoint_dir = "{}/{}/{}/checkpoint".format(run_root,args.model_name, args.dataname)
    args.metic_dir = "{}/{}/{}/".format(run_root,args.model_name, args.dataname)

    if args.checkpoint_dir is not None:
        if not os.path.exists(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir)