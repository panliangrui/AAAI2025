import numpy as np
import torch
from tqdm import tqdm
from Lightning.BagDataset_GPU import load_dataset
from Lightning.Tensorboard_LG import tensorboard_lg
from Lightning.checkpoint_LG import checkpoint_lg
# from Lightning.metric_LG import metric_lg
from Lightning.metric_LG_new import metric_lg
from Lightning.stop_early_LG import stop_early_lg
from torch.optim import Adam
from torch import nn
from Model.model_loader import load_model
from club.ranger import Ranger



class Lightning():
    def __init__(self,
                 args
                ):
        self.melg=metric_lg(metric_dir=args.metic_dir)
        self.stlg=stop_early_lg(metric=args.metric, patient=args.patient)
        self.tlg=tensorboard_lg(tensorboard_folder=args.tensorboard_dir)
        self.cklg=checkpoint_lg(metric=args.metric, checkpoint_dir=args.checkpoint_dir)

        self.args=args

    def get_current_result(self):
        return {"acc":0,"loss":0}
    def inference(self, bag_label, bag_feats, model, optimzier=None, train=False):
        if train:
            prob, loss = model(bag_feats,bag_label)

            optimzier.zero_grad()
            loss.backward()
            optimzier.step()

        else:
            with torch.no_grad():
                prob, loss = model(bag_feats,bag_label)
                prob = torch.sigmoid(prob)
            return prob

    def train(self,):
        model = load_model(self.args)
        optimizer = Ranger(model.parameters(), lr=self.args.lr, weight_decay=self.args.wd)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.args.num_epochs, 0.000005)

        train_dataloder = load_dataset(train="train", args=self.args)
        val_dataloader = load_dataset(train="val", args=self.args)


        model.to(self.args.device)
        self.tlg.init_tensorbard(self.args.seed)

        for epoch in range(self.args.start_epoch, self.args.num_epochs):
            current_result = self.get_current_result()
            model.train()

            for bag_label, bag_feats in tqdm(train_dataloder, desc="{} {} Seed {} Training {}".format(self.args.model_name,
                                                                                                     self.args.dataname,
                                                                                                     self.args.seed,epoch)):
                self.inference(bag_label, bag_feats, model, optimizer, train=True)

            self.tlg.refresh_log(epoch=epoch, recoder_dict=current_result, step=len(train_dataloder))
            scheduler.step()

            current_result = self.val(model,val_dataloader)
            print(current_result)

            # 保存模型
            self.cklg.save_epoch_checkpoint(
                epoch_metric=current_result[self.args.metric],
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                ckp_dir=self.cklg.checkpoint_dir,
                name=self.args.model_name,
                epoch_frq=self.args.epoch_frq,
                seed=self.args.seed)

            if self.stlg.stop(current_result[self.args.metric]):
                # 保存模型
                self.cklg.save_epoch_checkpoint(
                    epoch_metric=current_result[self.args.metric],
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    ckp_dir=self.cklg.checkpoint_dir,
                    name=self.args.model_name,
                    epoch_frq=self.args.epoch_frq,
                    seed=self.args.seed,
                    Last_epoch=True)
                break


    def val(self,model,val_dataloader,epoch=0,csv_path=None):
        model.eval()
        val_labels = []
        val_predictions = []
        with torch.no_grad():
            for bag_label, bag_feats in val_dataloader:
                probs = self.inference(bag_label, bag_feats, model)

                val_labels.extend(bag_label.squeeze(0).cpu().numpy())
                val_predictions.extend(probs.cpu().numpy())


        test_labels = np.array(val_labels)
        test_predictions = np.array(val_predictions)
        current_result=self.melg.get_reslut(test_predictions, test_labels,epoch=epoch,csv_path=csv_path)

        # current_result = self.melg.get_reslut(epoch, test_predictions, test_pred_labels, test_labels, csv_path)

        return current_result

    def test(self,
             epoch,
             checkpoint_path=None,
             csv_path=None,
             ):

        model = load_model(self.args)
        if checkpoint_path is not None:
            self.cklg.load_checkpoint(model=model, path=checkpoint_path)
        model.to(self.args.device)
        test_dataloader = load_dataset(train="test", args=self.args)
        test_score=self.val(model, test_dataloader, epoch, csv_path)

        return test_score


