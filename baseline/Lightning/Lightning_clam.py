import numpy as np
import torch
from tqdm import tqdm
from Lightning.BagDataset_GPU import load_dataset
from Lightning.Tensorboard_LG import tensorboard_lg
from Lightning.checkpoint_LG import checkpoint_lg
from Lightning.metric_LG import metric_lg
# from Lightning.metric_LG_new import metric_lg
from Lightning.stop_early_LG import stop_early_lg
from torch.optim import Adam
from torch import nn
from baseline.Models.model_loader import load_model



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

    def inference(self, bag_label, bag_feats, model, optimzier=None, current_result=None, train=False):
        if train:
            result_dict = model(bag_feats,bag_label)

            optimzier.zero_grad()
            result_dict["loss"].backward()
            optimzier.step()
        else:
            with torch.no_grad():
                result_dict = model(bag_feats,bag_label)
                probs = result_dict["probs"]

            return probs

    def train(self,):
        model = load_model(self.args)
        optimizer = Adam(model.parameters(), lr=self.args.lr, weight_decay=self.args.wd)
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
                self.inference(bag_label.squeeze(0), bag_feats.squeeze(0), model, optimizer, current_result, train=True)

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


    def val(self,model,val_dataloader):

        model.eval()

        val_labels = []
        val_predictions = []
        val_pred_labels = []

        with torch.no_grad():
            for bag_label, bag_feats in val_dataloader:
                probs = self.inference(bag_label.squeeze(0), bag_feats.squeeze(0), model,None,None,False)

                val_labels.extend(bag_label.squeeze(0).cpu().numpy())
                val_predictions.extend(probs.cpu().numpy())
                val_pred_labels.extend(np.argmax(probs.cpu().numpy(), axis=1))

        test_labels = np.array(val_labels)
        test_predictions = np.array(val_predictions)
        test_pred_labels = np.array(val_pred_labels)


        current_result=self.melg.get_reslut(0,test_predictions,test_pred_labels, test_labels)

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
        model.eval()
        test_dataloader = load_dataset(train="test", args=self.args)

        # 初始化变量用于统计
        predictions, pred_labels, true_labels = [], [], []

        for bag_label, bag_feats in tqdm(test_dataloader, desc="Testing"):

            prob = self.inference(bag_label.squeeze(0), bag_feats.squeeze(0), model,None,None,False)
            predictions.extend(prob.data.cpu().numpy())
            true_labels.extend(bag_label.squeeze(0).cpu().numpy())
            pred_labels.extend(np.argmax(prob.data.cpu().numpy(), axis=1))

        predictions = np.array(predictions)
        pred_labels = np.array(pred_labels)
        true_labels = np.array(true_labels)

        # test_score=self.melg.get_reslut(predictions, true_labels,epoch=epoch,csv_path=csv_path)

        test_score = self.melg.get_reslut(epoch, predictions, pred_labels, true_labels, csv_path)
        return test_score


