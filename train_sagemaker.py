import os
import time
# from scripts.freeze_graph import freeze_graph
# from inference.kv_model import KVModel
# from inference.generic_util import glob_folder, read_image_list

from sagemaker.pytorch import PyTorch

instance_type = 'ml.p2.xlarge'
train_data_path = 's3://research-tw/Lin/data_version2'
output_path = 's3://research-tw/Lin/output'
# code_location = 's3://rnd-ocr/JeffYang/Layout/code/'
role = "arn:aws:iam::533155507761:role/service-role/AmazonSageMaker-ExecutionRole-20190312T160681"
checkpoint_path='s3://research-tw/Lin/output'
source_dir = "."

pytorch_estimator = PyTorch(entry_point='run_classifier.py',
                            checkpoint_s3_uri=checkpoint_path,
                            source_dir=source_dir,
                            output_path=output_path,
                            role=role,
                            train_instance_type=instance_type,
                            train_instance_count=1,
                            train_volume_size=200,
                            base_job_name= 'Lin-Sompo',
                            train_max_run=5*86400,  # 86400s ~ 1day
                            framework_version='1.0.0',
                            py_version="py3",
                            train_use_spot_instances=True,
                            train_max_wait=5*86400 + 300,
                            hyperparameters={"sagemaker": "True",
                                             "backbone": "SelfAttention_GCN_backbone"}
                            )

pytorch_estimator.fit({"train": train_data_path})