import train_test
from Config import args_base
from Config.base import *
from Model import nystrom_attention


def train(dataname,model_name):
    args = args_base.get_argp(dataname=dataname, model_name=model_name)
    run_root = f"./valine_auc_best"
    reset_run_root(args, run_root)

    # args.start_seed=4
    train_test.main(args, train=True, test=True, best=True)

def main(model_name):
    train("xiangya3", model_name)
    train("TCGA", model_name)
    # train("xiangya2", model_name)

if __name__ == '__main__':
    # main("maxpooling")
    # main("meanpooling")
    # main("abmil")
    # main("dsmil")
    # main("clam_sb")
    main("clam_mb")
    main("transmil")
    # main("ilra")
    # main("wikg")



