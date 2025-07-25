
def load_model(args):
    from Model.mvmpmil import MvMpMIL
    milnet = MvMpMIL(in_features=args.input_dim,num_classes=args.nclass).cuda()
    return milnet