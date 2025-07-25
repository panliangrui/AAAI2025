import os
from os.path import join


root="/root/autodl-tmp/plr_sata_aaai/"
def set_data(args):
    args.train_excel_path = os.path.join(root, "data", args.dataname, f"fold{args.seed}_train.csv")
    args.val_excel_path = os.path.join(root, "data", args.dataname, f"fold{args.seed}_val.csv")
    args.test_excel_path = os.path.join(root, "data", args.dataname, f"fold{args.seed}_test.csv")
    args.data_path = f"/root/autodl-tmp/Dataset/{args.dataname}/features_gigapath/"

    args.n_class=2
    args.task = "multiclass"

    args.ckpt_dir="recoder/{}/checkpoint".format(args.dataname)
    args.ckp=join(args.ckpt_dir,'seed{}_checkpoint-best.pth'.format(args.seed))

    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)


    args.csv_path="recoder/{}/test_result.csv".format(args.dataname)