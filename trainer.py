import os
import pickle
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
from time import time
from torchvision import transforms
from utils.torch_utils import Ranger, FocalLoss
from utils.avgmeter import AverageMeter
from collections import defaultdict


INFO_DICT = {
    "formal": defaultdict(lambda: {"c": 0, "w": 0}),
    "final": defaultdict(lambda: {"c": 0, "w": 0}),
    "classification": defaultdict(lambda: {"c": 0, "w": 0}),
    }

LOGGER = logging.getLogger("__main__")

def print_vars(input_):
    for key, value in input_.items():
        LOGGER.info(f"{key:20}: {value}")

def compute_time(func):
    def decorated_func(*args, **kwargs):
        start = time()
        ret = func(*args, **kwargs)
        end = time()
        # LOGGER.info(f"spent: {end - start:.3f} s")

        return ret, end - start

    return decorated_func

def accuracy(pred, y):
    return ((pred == y).float().mean(0).cpu().item(), pred.shape[0])

def recall_precision(pred, y, class_):
    """
    calulate the accuracy and recall for class_
    """
    class_pred = pred == class_
    class_y = y == class_

    correct = ((pred == y)[class_pred]).sum().cpu().item()

    num_class = class_y.sum().cpu().item()
    num_pred = class_pred.sum().cpu().item()

    # pair: (value, num)
    if num_class == 0:
        recall_pair = (0, 0)
    else:
        recall_pair = (correct / num_class, num_class)

    if num_pred == 0:
        precision_pair = (0, 0)
    else:
        precision_pair = (correct / num_pred, num_pred)

    # LOGGER.info(precision_pair)

    return recall_pair, precision_pair

class Trainer():
    def __init__(self, hparams, model):
        self.hparams = hparams
        self.model = model

        self.device = hparams["device"]
        
        # adjust the ratio of background class
        hparams["ratio_focal_formal"][0] = hparams["ratio_focal_formal"][0] * 15
        hparams["ratio_focal_kv"][0] = hparams["ratio_focal_kv"][0] * 15

        ## set up loss founction
        self.criterion_formal = FocalLoss(alpha=hparams["ratio_focal_formal"])
        self.criterion_normal = FocalLoss(alpha=hparams["ratio_focal_kv"])

        self.criterion_classification = nn.CrossEntropyLoss(
            weight=torch.from_numpy(np.array(hparams["ratio_focal_category"])).float().to(self.device))

        self.model.to(self.device)

        self.optimizer = Ranger(self.model.parameters(), lr=hparams["lr"], weight_decay=hparams["weight_decay"])

        print_vars(hparams)

        LOGGER.info(self.device)

    def adjust_learning_rate(self, initial_lr, optimizer, epoch):
        # lr = initial_lr * (0.9**(epoch // 10))
        lr = initial_lr * (0.1 ** (epoch // 5))
        LOGGER.info(f"linear decay... current lr: {lr}")
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def run_iteration(self, batch, model):
        self.model.train()

        image, vertexes, A, labels, category = batch

        image = image.float().to(self.device)
        V = vertexes.float().to(self.device)
        A = A[:, :, :4, :].float().to(self.device)

        y_train, category_train = labels.long().to(self.device), category.long().to(self.device)

        y_train_formal = y_train[0][:, 0]
        y_train_final = y_train[0][:, 1]

        output_formal, output_final, output_classification = self.model(image, V, A, None)

        loss_formal = self.criterion_formal(output_formal[0],
                                            y_train_formal)
        loss_final = self.criterion_normal(output_final[0],
                                            y_train_final)

        loss_classification = self.criterion_classification(output_classification,\
                                                            category_train)

        loss = self.hparams["lambda_kv"] * (loss_formal * 0.5 + loss_final) + loss_classification

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        formal_pred = torch.argmax(F.softmax(output_formal[0], -1), -1)
        final_pred = torch.argmax(F.softmax(output_final[0], -1), -1)

        classification_pred = torch.argmax(F.softmax(output_classification, -1), -1)

        # evaluate the result per class
        formal_rp_pair = [recall_precision(formal_pred, y_train_formal, class_) \
            for class_ in range(len(self.hparams["key_index"]))]
        
        final_rp_pair = [recall_precision(final_pred, y_train_final, class_) \
            for class_ in range(len(self.hparams["key_index"]) * 2 + 1)]

        result_dict = {
            "loss": (loss.cpu().item(), output_classification.shape[0]),
            "loss_formal": (loss_formal.cpu().item(), output_classification.shape[0]),
            "loss_final": (loss_final.cpu().item(), output_classification.shape[0]),
            "loss_class": (loss_classification.cpu().item(), output_classification.shape[0]),
            "class_acc": accuracy(classification_pred, category_train)
        }

        for class_, pair_ in enumerate(list(zip(*formal_rp_pair))[0]):
            result_dict[f"formal_r_{class_}"] = pair_
        
        for class_, pair_ in enumerate(list(zip(*formal_rp_pair))[1]):
            result_dict[f"formal_p_{class_}"] = pair_

        for class_, pair_ in enumerate(list(zip(*final_rp_pair))[0]):
            result_dict[f"final_r_{class_}"] = pair_

        for class_, pair_ in enumerate(list(zip(*final_rp_pair))[1]):
            result_dict[f"final_p_{class_}"] = pair_

        return result_dict

    @compute_time
    def run_epoch(self, train_loader, model):
        result_meters = defaultdict(AverageMeter)

        for batch in train_loader:
            result_dict = self.run_iteration(batch, model)

            for key, value in result_dict.items():
                result_meters[key].update(*value)

        return result_meters

    def evaluate_iteration(self, batch, model):
        self.model.eval()

        image, vertexes, A, labels, category = batch

        image = image.float().to(self.device)
        V = vertexes.float().to(self.device)
        A = A[:, :, :4, :].float().to(self.device)

        y_train, category_train = labels.long().to(self.device), category.long().to(self.device)

        y_train_formal = y_train[0][:, 0]
        y_train_final = y_train[0][:, 1]

        output_formal, output_final, output_classification = self.model(image, V, A, None)

        loss_formal = self.criterion_formal(output_formal[0],
                                            y_train_formal)
        loss_final = self.criterion_normal(output_final[0],
                                           y_train_final)

        loss_classification = self.criterion_classification(output_classification,\
                                                            category_train)

        loss = self.hparams["lambda_kv"] * (loss_formal * 0.5 + loss_final) + loss_classification

        formal_pred = torch.argmax(F.softmax(output_formal[0], -1), -1)
        final_pred = torch.argmax(F.softmax(output_final[0], -1), -1)

        classification_pred = torch.argmax(F.softmax(output_classification, -1), -1)

        # evaluate the result per class
        formal_rp_pair = [recall_precision(formal_pred, y_train_formal, class_) \
            for class_ in range(len(self.hparams["key_index"]))]
        
        final_rp_pair = [recall_precision(final_pred, y_train_final, class_) \
            for class_ in range(len(self.hparams["key_index"]) * 2 + 1)]

        result_dict = {
            "loss": (loss.cpu().item(), output_classification.shape[0]),
            "loss_formal": (loss_formal.cpu().item(), output_classification.shape[0]),
            "loss_final": (loss_final.cpu().item(), output_classification.shape[0]),
            "loss_class": (loss_classification.cpu().item(), output_classification.shape[0]),
            "class_acc": accuracy(classification_pred, category_train)
        }

        for class_, pair_ in enumerate(list(zip(*formal_rp_pair))[0]):
            result_dict[f"formal_r_{class_}"] = pair_
        
        for class_, pair_ in enumerate(list(zip(*formal_rp_pair))[1]):
            result_dict[f"formal_p_{class_}"] = pair_

        for class_, pair_ in enumerate(list(zip(*final_rp_pair))[0]):
            result_dict[f"final_r_{class_}"] = pair_

        for class_, pair_ in enumerate(list(zip(*final_rp_pair))[1]):
            result_dict[f"final_p_{class_}"] = pair_

        return result_dict

    @compute_time
    @torch.no_grad()
    def evaluate(self, val_loader, model):
        result_meters = defaultdict(AverageMeter)

        for batch in val_loader:
            result_dict = self.evaluate_iteration(batch, model)

            for key, value in result_dict.items():
                result_meters[key].update(*value)

        return result_meters

    def print_log(self, epoch, spent_time, meter_dict, **kwargs):
        LOGGER.info(f"epoch: {epoch}, spent {spent_time:.4f} s, loss: {meter_dict['loss'].avg:.4f}\n")

        LOGGER.info("[final]")
        LOGGER.info(f"loss: {meter_dict['loss_final'].avg:.4f}")

        avg_dict = defaultdict(AverageMeter)

        for i in range(len(self.hparams["key_index"]) * 2 + 1):
            recall_key = f"final_r_{i}"
            precision_key = f"final_p_{i}"
            LOGGER.debug(f"class {i:3}, recall {meter_dict[recall_key].avg:.3f} ({meter_dict[recall_key].count:3}), "
                    f"precision {meter_dict[precision_key].avg:.3f} ({meter_dict[precision_key].count:3})")

            avg_dict["r"].update(meter_dict[recall_key].avg, meter_dict[recall_key].count > 0)
            avg_dict["p"].update(meter_dict[precision_key].avg, meter_dict[precision_key].count > 0)

        LOGGER.info(f"avg recall: {avg_dict['r'].avg:.3f}, avg precision: {avg_dict['p'].avg:.3f}\n")

        LOGGER.info("[classification]")

        if kwargs.get("best_acc", None):
            LOGGER.info(f"loss: {meter_dict['loss_class'].avg:.3f}, acc: {meter_dict['class_acc'].avg:.3f} "
              f"/ {kwargs['best_acc']:.3f} ({meter_dict['class_acc'].count:3})")
        else:
            LOGGER.info(f"loss: {meter_dict['loss_class'].avg:.3f}, acc: {meter_dict['class_acc'].avg:.3f} "
                f"({meter_dict['class_acc'].count:3})")

        LOGGER.info("-" * 30)

    def fit(self, train_loader, val_loader=None):

        LOGGER.info(f"{len(train_loader)}, {len(val_loader)}")
        
        best_acc = 0

        for epoch in range(1, self.hparams["epochs"] + 1):
            if self.hparams["linear_decay"] == "True":
                self.adjust_learning_rate(self.hparams["lr"], self.optimizer, epoch % 100)

            train_meters, spent_time = self.run_epoch(train_loader, self.model)

            self.print_log(epoch, spent_time, train_meters)
           
            if epoch % 5 == 0:
                eval_meters, spent_time = self.evaluate(val_loader, self.model)

                if eval_meters["class_acc"].avg >= best_acc:
                    best_acc = eval_meters["class_acc"].avg
                    self._save(-1)

                LOGGER.info("="*25 + "eval" + "="*25)

                self.print_log(epoch, spent_time, eval_meters, best_acc=best_acc)

                LOGGER.info("="*23 + "eval end" + "="*23)

    def test(self, val_loader):
        eval_meters, spent_time = self.evaluate(val_loader, self.model)
        self.print_log(epoch, spent_time, eval_meters)

    def _save(self, epoch):
        if not os.path.exists(self.hparams["checkpoints_folder"]):
            os.makedirs(self.hparams["checkpoints_folder"])
        path = os.path.join(self.hparams["checkpoints_folder"], f"model_{epoch}.pth")
        torch.save(
            self.model.state_dict(),
            path)

        with open(os.path.join(self.hparams["checkpoints_folder"], "key_index.pickle"), "wb") as f:
            pickle.dump(self.hparams["key_index"], f)

        with open(os.path.join(self.hparams["checkpoints_folder"], "tok_to_id.pickle"), "wb") as f:
            pickle.dump(self.hparams["tok_to_id"], f)
            
        with open(os.path.join(self.hparams["checkpoints_folder"], "class_index.pickle"), "wb") as f:
            pickle.dump(self.hparams["class_index"], f)



