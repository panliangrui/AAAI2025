
def load_model(args):
    if args.model_name == 'dsmil':
        from baseline.Models import dsmil as mil
        i_classifier = mil.FCLayer(in_size=args.input_dim, out_size=args.nclass).cuda()
        b_classifier = mil.BClassifier(input_size=args.input_dim, output_class=args.nclass).cuda()
        milnet = mil.MILNet(i_classifier, b_classifier).cuda()
    elif args.model_name == 'abmil':
        from baseline.Models.abmil import Attention
        milnet = Attention(in_size=args.input_dim, out_size=args.nclass).cuda()
    elif args.model_name == "wikg":
        from baseline.Models.WiKG import WiKG
        milnet = WiKG(dim_in=args.input_dim,n_classes=args.nclass).cuda()
    elif args.model_name == "ilra":
        from baseline.Models.ILRA import ILRA
        milnet = ILRA(feat_dim=args.input_dim,n_classes=args.nclass).cuda()
    elif args.model_name == "maxpooling":
        from baseline.Models.maxpooling_meanpooling import Maxpooling
        milnet = Maxpooling(in_size=args.input_dim,n_classes=args.nclass).cuda()
    elif args.model_name == "meanpooling":
        from baseline.Models.maxpooling_meanpooling import Meanpooling
        milnet = Meanpooling(in_size=args.input_dim,n_classes=args.nclass).cuda()
    elif args.model_name =="clam_sb":
        from baseline.Models.clam import CLAM_SB
        milnet = CLAM_SB().cuda()
    elif args.model_name =="clam_mb":
        from baseline.Models.clam import CLAM_MB
        milnet = CLAM_MB().cuda()
    elif args.model_name =="transmil":
        from baseline.Models.TransMIL import TransMIL
        milnet = TransMIL(input_dim=args.input_dim,n_classes=args.nclass).cuda()
    return milnet