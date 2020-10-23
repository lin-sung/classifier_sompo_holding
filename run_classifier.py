import os
import torch
import logging
import argparse
from trainer import Trainer
from data_loader import DataModule

from models.gcn import Attention_GCN_backbone, Jeff_Attention_GCN_backbone, SelfAttention_GCN_backbone, \
    SelfAttention_GCN_classifier, MultiAttention_GCN_backbone

LOGGER = logging.getLogger("__main__")

LOGGER.setLevel(logging.DEBUG)

LOGGER.addHandler(logging.StreamHandler())

def generate_parser():
    parser = argparse.ArgumentParser(description="MSAU")
    parser.add_argument("--test", action="store_true",
                        help="whether to use testing mode, or using train mode")
    parser.add_argument("--sagemaker", type=str, default="False", 
                        choices=["True", "False"], help="whether use in sagemaker")
    parser.add_argument("--train_data_dir", type=str,
                        help="the directory of the train data")
    parser.add_argument("--test_data_dir", type=str,
                        help="the directory of the test data")
    parser.add_argument("--checkpoints_folder", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=100,
                        help="maximum epochs")
    parser.add_argument("--backbone", type=str, default="Attention_GCN_backbone")

    parser.add_argument("--checkpoints_dir", default="", type=str)

    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--cuda", default=False, action="store_true")
    parser.add_argument("--linear_decay", default="False", choices=["True", "False"], type=str)
    parser.add_argument("--lambda_kv", type=float, default=1)
    parser.add_argument("--preprocessing", default="False", choices=["True", "False"], type=str)
    parser.add_argument("--extend_dataset", default="", type=str)

    return parser.parse_args()

def main():
    args = generate_parser()
    hparams = vars(args) # transform to dict, which is easy to save

    if hparams["sagemaker"] == "True":
        hparams["train_data_dir"] = os.environ['SM_CHANNEL_TRAIN']
        hparams["checkpoints_folder"] = os.environ['SM_MODEL_DIR']
        hparams["cuda"] = True

    device = torch.device("cuda" if torch.cuda.is_available() and hparams["cuda"] else "cpu")
    hparams["device"] = device
    
    if not hparams["test"]:
        if not os.path.exists(hparams["checkpoints_folder"]):
            os.makedirs(hparams["checkpoints_folder"])
        LOGGER.addHandler(logging.FileHandler(os.path.join(hparams["checkpoints_folder"], "train.log"), "w"))
        # training
        data_module = DataModule(hparams["train_data_dir"], 1, hparams["extend_dataset"], 
            eval(hparams["preprocessing"]), 2)

        train_loader = data_module.get_train_loader()
        val_loader = data_module.get_val_loader()
        
        add_list = [int, float, str, list]
        for k, v in vars(data_module).items():
            is_add = False
            for type_ in add_list:
                is_add = is_add or isinstance(v, type_)

            if is_add:
            # if isinstance(v, int) or v is float or v is str:
                hparams[k] = v
            
        
        hparams["key_index"] = data_module.key_index
        hparams["class_index"] = data_module.class_index
        hparams["tok_to_id"] = data_module.tok_to_id

        backbone = eval(hparams["backbone"])

        model = backbone(len(data_module.tok_to_id) + 4, 4, 
            class_formal_key=len(data_module.key_index) + 1, key_index=data_module.key_index, num_classification=3)

        if hparams["checkpoints_dir"] != "":
            model.load_state_dict(torch.load(hparams["checkpoints_dir"], map_location="cpu"))
            LOGGER.info("Successfully load checkpoints...")

        trainer = Trainer(hparams, model)
        trainer.fit(train_loader, val_loader)

    else:
        # testing
        data_module = DataModule(hparams["train_data_dir"], 1, 2)

        val_loader = data_module.get_val_loader()
        
        add_list = [int, float, str, list]
        for k, v in vars(data_module).items():
            is_add = False
            for type_ in add_list:
                is_add = is_add or isinstance(v, type_)

            if is_add:
            # if isinstance(v, int) or v is float or v is str:
                hparams[k] = v

        backbone = eval(hparams["backbone"])
        model = backbone(len(data_module.tok_to_id) + 4, 4, 
            class_formal_key=len(data_module.key_index) + 1, key_index=data_module.key_index, num_classification=3)

        model.load_state_dict(torch.load(hparams["checkpoints_dir"], map_location="cpu"))

        trainer = Trainer(hparams, model)
        trainer.test(val_loader)

if __name__ == "__main__":
    main()
