import os.path
from Config.base import *

def main(args,train,test,best):

    for i in range(args.start_seed,args.end_seed):
        args.seed = i
        set_data(args)
        if "clam" in args.model_name:
            from baseline.Lightning.Lightning_clam import Lightning
        else:
            from baseline.Lightning.Lightning import Lightning

        lg = Lightning(args)
        if train:
            lg.train()

        if test:
            if best:
                best_ckp = os.path.join(args.checkpoint_dir,
                                              "{}_best_seed{}.pth".format(args.model_name, args.seed))
            else:
                best_ckp = os.path.join(args.checkpoint_dir,
                                              "{}_Last_seed{}.pth".format(args.model_name, args.seed))
            lg.test(epoch=0 if i == 0 else 200,
                    checkpoint_path=best_ckp,
                    csv_path=args.test_best_csv_path if best else args.test_last_csv_path)