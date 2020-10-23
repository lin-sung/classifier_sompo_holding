import os
import json
import glob
import random
import pickle
import logging
from pathlib import Path
import numpy as np
from .data_utils import load_msau_corpus, load_msau_classes, load_json, \
    load_path_list, load_samples, load_path_list

import torch
from torch.utils.data import DataLoader, Dataset, Subset

import sys

sys.path.append("..")

from utils.utils import to_categorical, get_iou

from collections import Counter, defaultdict

from PIL import Image

from torchvision import transforms

from utils.graph import Graph

LOGGER = logging.getLogger("__main__")


# LABEL_DICT = {
# 'total_insurance_burden_score': 'total_insurance_burden_score', 
# 'receipt_amount': 'receipt_amount', 
# 'copayment_ratio': 'copayment_ratio', 
# 'date': 'date', 
# 'client_name': 'client_name', 
# 'insurance_covered_fee': 'insurance_covered_fee', 
# 'total_non_insurance_burden': 'total_non_insurance_burden', 
# 'department_name': 'department_name', 
# 'patient_status': 'patient_status', 
# 'hospital_name': 'hospital_name', 
# 'psychiatric_specialty_therapy': 'psychiatric_specialty_therapy', 
# 'room_charge': 'room_charge', 
# 'total_insurance_burden': 'total_insurance_burden', 
# 'receipt_stamp_circle': 'stamp', 
# 'receipt_stamp': 'stamp', 
# 'hospital_stamp': 'stamp', 
# 'total_insurance_burden_score-insurance_covered_fee': 'insurance_covered_fee', 
# 'total_insurance_burden_score-total_non_insurance_burden': 'total_non_insurance_burden', 
# 'total_insurance_burden-insurance_covered_fee': 'insurance_covered_fee', 
# 'total_insurance_burden-total_non_insurance_burden': 'total_non_insurance_burden', 
# 'total_non_insurance_burden-total_insurance_burden_score': 'total_insurance_burden_score', 
# 'insurance_covered_fee-total_non_insurance_burden': 'total_non_insurance_burden', 
# 'total_insurance_burden_score-insurance_covered_fee-total_non_insurance_burden': 'total_non_insurance_burden', 
# 'receipt_stamp_square': 'stamp', 
# 'total_insurance_burden-total_insurance_burden_score': 'total_insurance_burden_score', 
# 'total_non_insurance_burden-total_insurance_burden_score-total_insurance_burden': 'total_insurance_burden', 
# 'total_insurance_burden-insurance_covered_fee-total_non_insurance_burden': 'total_non_insurance_burden', 
# 'total_non_insurance_burden-insurance_covered_fee': 'insurance_covered_fee', 
# 'total_insurance_burden_score-total_insurance_burden-total_non_insurance_burden': 'total_non_insurance_burden', 
# 'not be admitted': 'patient_status'}

LABEL_DICT = {
'date': 'date', 
'patient_status': 'patient_status', 
'not be admitted': 'patient_status'}

def to_bag_of_words(x, prob, tok_to_id, blank_idx):
    n_token = len(tok_to_id)
    output = np.zeros((n_token))
        
    for i in range(len(x)):
        index = tok_to_id.get(x[i], blank_idx)
        # temp[index] += 1
        if np.random.binomial(size=1, n=1, p=prob):
            index = random.choice(range(n_token))

        output[index] += 1

    return output


def get_classification_label(label):

    patient_status = "others"
    patient_string = ""

    for info in label["attributes"]["_via_img_metadata"]["regions"]:
        region_attributes = info["region_attributes"]

        # construct the class
        if region_attributes["formal_key"] == "patient_status" and \
            region_attributes["key_type"] == "value":

            # # process the category to int
            # if "入院" in region_attributes["label"]:
            #     # category = "入院"
            #     patient_status = "be admitted"

            # elif "外来" in region_attributes["label"]:
            #     # category = "外来"
            #     patient_status = "not be admitted"
            # else:
            #     # category = "others"
            #     patient_status = "others"
            patient_string += region_attributes["label"]

            # process the category to int
    if "入院" in patient_string:
        # category = "入院"
        patient_status = "be admitted"

    elif "外来" in patient_string:
        # category = "外来"
        patient_status = "not be admitted"
    else:
        # category = "others"
        patient_status = "others"

    for info in label["attributes"]["_via_img_metadata"]["regions"]:
        region_attributes = info["region_attributes"]

        if region_attributes["note"].strip() in ["be admitted", "not be admitted"]:

            patient_status = region_attributes["note"]

            # print(region_attributes["formal_key"])

    return patient_status

class TextNoise:
    """Add noise to the text"""

    def __init__(self, tok_to_id, prob):
        self.tok_to_id = tok_to_id
        self.prob = prob
        self.n_token = len(self.tok_to_id)

    def __call__(self, x):
        # temp = np.zeros((self.n_token))
        output = np.zeros((self.n_token))
        
        for i in range(len(x)):
            index = self.tok_to_id[x[i]]
            # temp[index] += 1
            if np.random.binomial(size=1, n=1, p=self.prob):
                index = random.choice(range(self.n_token))

            output[index] += 1

        # LOGGER.info(output.max())

        # LOGGER.info(np.abs(temp - output).sum(), np.abs(temp - output).sum() / len(x) * 100)

        return output

class TextToTensor:
    def __init__(self, max_value):
        self.max_value = max_value

    def __call__(self, x):
        x = x / self.max_value
        return torch.from_numpy(x)

class BoxNoise:
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, x):
        width = x[2] - x[0]
        height = x[3] - x[1]
        
        left = np.random.uniform(0, width * self.scale)
        right = np.random.uniform(0, width * self.scale)

        top = np.random.uniform(0, height * self.scale)
        bottom = np.random.uniform(0, height * self.scale)

        ret = np.zeros(len(x))

        ret[0] = max(ret[0], x[0] - left)
        ret[1] = max(ret[1], x[1] - top)

        ret[2] = max(ret[2], x[2] + right)
        ret[3] = max(ret[3], x[3] + bottom)

        return ret

class TransformSubset(Dataset):
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    def __init__(self, dataset, indices, transform=None, text_transform=None, box_transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
        self.text_transform = text_transform
        self.box_transform = box_transform

    def __getitem__(self, idx):
        index = self.indices[idx]

        image = self.dataset[index]["image"]
        category = self.dataset[index]["category"]
        texts = self.dataset[index]["texts"]

        labels = self.dataset[index]["labels"].astype(int)
        A = self.dataset[index]["A"].astype(float)
        vertexes = self.dataset[index]["vertexes"].astype(float)
        
        if self.text_transform is not None:
            for i, text in enumerate(texts):
                vertexes[i, 4:] = self.text_transform(text)

        if self.box_transform is not None:
            for i in range(vertexes.shape[0]):
                vertexes[i, :4] = self.box_transform(vertexes[i, :4])

        # if self.transform is not None:
        #     image = self.transform(image)
        
        # if self.text_transform is not None:
        #     text = self.text_transform(text)

        return image, vertexes, A, labels, category

    def __len__(self):
        return len(self.indices)


class DataModule():
    def __init__(self, data_dir, batch_size, extend_dataset="", preprocessing=False, num_workers=16):
        self.data_dir = data_dir
        # self.corpus_dir = os.path.join(data_dir, "corpus.json")
        # self.classes_dir = os.path.join(data_dir, "classes.json")
        # self.key_types_dir = os.path.join(data_dir, "key_types.json")
        self.samples_dir = os.path.join(data_dir, "images")
        self.labels_dir = os.path.join(data_dir, "labels")
        # self.test_samples_dir = os.path.join(data_dir, "test_samples")
        self.corpus_dir = os.path.join(data_dir, "corpus_782.json")
        
        self.train_split = os.path.join(self.data_dir, "train.lst")
        self.val_split = os.path.join(self.data_dir, "val.lst")
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.extend_dataset = extend_dataset

        self.is_training = True # dummy

        if extend_dataset != "":
            processed_data_file = os.path.join(self.data_dir, "ocr_extend_processed_data.pickle")
        else:
            processed_data_file = os.path.join(self.data_dir, "processed_data.pickle")

        key_index_file = os.path.join(self.data_dir, "key_index.pickle")
        tok_to_id_file = os.path.join(self.data_dir, "tok_to_id.pickle")
        class_index_file = os.path.join(self.data_dir, "class_index.pickle")

        if preprocessing or not os.path.exists(processed_data_file):

            LOGGER.info("process data...")

            self.class_index = {
                "others": 0,
                "not be admitted": 1,
                "be admitted": 2
            }
            self.corpus = load_msau_corpus(self.corpus_dir)

            self.data, self.key_index, self.ratio_focal_kv, self.ratio_focal_formal, \
                self.ratio_focal_category = self._read_data()

            if extend_dataset != "":
                self._ocr_extend_dataset(self.data, extend_dataset)

            self.tok_to_id, self.id_to_tok, self.n_token, self.corpus, self.blank_idx = \
                self._process_corpus(self.corpus, self.data)

            self._process_data(self.data)
            with open(processed_data_file, "wb") as f:
                save = {
                    "data": self.data, 
                    # "key_index": self.key_index, 
                    "ratio_focal_kv": self.ratio_focal_kv, 
                    "ratio_focal_formal": self.ratio_focal_formal, 
                    # "tok_to_id":  self.tok_to_id, 
                    "id_to_tok": self.id_to_tok, 
                    "n_token": self.n_token, 
                    "blank_idx": self.blank_idx,
                    "ratio_focal_category": self.ratio_focal_category
                }
                pickle.dump(save, f)

            with open(key_index_file, "wb") as f:
                pickle.dump(self.key_index, f)

            with open(tok_to_id_file, "wb") as f:
                pickle.dump(self.tok_to_id, f)
            
            with open(class_index_file, "wb") as f:
                pickle.dump(self.class_index, f)

            with open(os.path.join(self.data_dir, "new_corpus.json"), "w") as f:
                json.dump(self.corpus, f, ensure_ascii=False, sort_keys=True)

        else:
            with open(processed_data_file, "rb") as f:
                save = pickle.load(f)

                self.data = save["data"] 
                # self.key_index = save["key_index"] 
                self.ratio_focal_kv = save["ratio_focal_kv"] 
                self.ratio_focal_formal = save["ratio_focal_formal"] 
                # self.tok_to_id = save["tok_to_id"]  
                self.id_to_tok = save["id_to_tok"] 
                self.n_token = save["n_token"] 
                self.blank_idx = save["blank_idx"] 
                self.ratio_focal_category = save["ratio_focal_category"]

            with open(key_index_file, "rb") as f:
                self.key_index = pickle.load(f)

            with open(tok_to_id_file, "rb") as f:
                self.tok_to_id = pickle.load(f)
            
            with open(class_index_file, "rb") as f:
                self.class_index = pickle.load(f)

            with open(os.path.join(self.data_dir, "new_corpus.json"), "w") as f:
                json.dump("".join(self.tok_to_id.keys()), f, ensure_ascii=False, sort_keys=True)

    # def to_bag_of_words(self, x, prob):
    #     output = np.zeros((self.n_token))
            
    #     for i in range(len(x)):
    #         index = self.tok_to_id[x[i]]
    #         # temp[index] += 1
    #         if np.random.binomial(size=1, n=1, p=prob):
    #             index = random.choice(range(self.n_token))

    #         output[index] += 1

    #     return output

    def _ocr_extend_dataset(self, data, path):
        paths = glob.glob(f'{path}/**/ocr_output.json', recursive=True)

        for path_ in paths:
            with open(path_, "r") as f:
                _data = json.load(f)

            name = Path(path_).parent.stem

            # load ground truth
            gt_name = f"{name}.json"

            if gt_name not in data:
                continue

            gt_dict = data[gt_name]

            data_key = f"ocr_extend_{name}.json"

            texts = []
            bboxes_for_train = []
            input_bboxes = []

            keys_type = []
            formal_keys = []

            match = 0
            not_match = 0

            # _data is list
            for label in _data:
                text = str(label["text"])
                text = text.replace('\n', '').replace(' ', '')

                match_index = -1
                match_value = 0

                now_box_list = [*label["location"][0], *label["location"][2]]
                now_box = {key: value for key, value in zip(["x1", "y1", "x2", "y2"], now_box_list)}

                for i, bbox_for_train in enumerate(gt_dict["bboxes"]):
                    test_box = {key: value for key, value in zip(["x1", "y1", "x2", "y2"], bbox_for_train)}

                    # LOGGER.info(now_box, test_box)
                    iou = get_iou(now_box, test_box)

                    if iou > 0.5 and iou > match_value:
                        match_index = i
                        match_value = iou

                if match_index != -1:
                    keys_type.append(gt_dict["keys_type"][match_index])
                    formal_keys.append(gt_dict["formal_keys"][match_index])
                else:
                    # LOGGER.info("didn't find match")
                    keys_type.append("")
                    formal_keys.append("")

                # construct the graph
                input_bbox_format, bbox_for_train = [*now_box_list, 1, 1, text, None], now_box_list

                texts.append(text)
                bboxes_for_train.append(bbox_for_train)
                input_bboxes.append(input_bbox_format)

            data[data_key]["image_file"] = gt_dict["image_file"]
            data[data_key]["keys_type"] = keys_type
            data[data_key]["formal_keys"] = formal_keys
            data[data_key]["category"] = gt_dict["category"]
            # data[key]["text"] = "".join(texts)
            data[data_key]["texts"] = texts            
            data[data_key]["bboxes"] = bboxes_for_train
            data[data_key]["input_bboxes"] = input_bboxes

    def _process_corpus(self, corpus, data):
        word_count = Counter([])

        for key, data_ in data.items():
            # LOGGER.info(key)
            # LOGGER.info(data_.keys())
            texts = "".join(data_["texts"])
            # texts = data_["texts"]

            word_count += Counter(texts)

            # for c in texts:
            #     if c not in tok_to_id:
            #         tok_to_id[c] = last
            #         id_to_tok[last] = c
            #         last += 1

        for c, num in word_count.items():
            if c not in corpus and num > 20:
                corpus += c

        blank_idx = 1 # equal token $
        tok_to_id = {tok: idx for idx, tok in enumerate(corpus)}
        id_to_tok = {idx: tok for tok, idx in tok_to_id.items()}

        # update tok_to_id and id_to_tok from the text in the 
        # last = len(tok_to_id)

        n_token = len(tok_to_id)

        return tok_to_id, id_to_tok, n_token, corpus, blank_idx

    def transform_graph_input(self, shape_attr, text):
        if shape_attr['name'] == 'polygon':
            x1, y1, x2, y2 = min(shape_attr['all_points_x']), \
                             min(shape_attr['all_points_y']), \
                             max(shape_attr['all_points_x']), \
                             max(shape_attr['all_points_y'])
            x, y, w, h = x1, y1, x2 - x1, y2 - y1
        else:
            x, y, w, h = shape_attr['x'], shape_attr['y'], shape_attr[
                'width'], shape_attr['height']

        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)

        x_1 = x + w
        y_1 = y + h

        return [x, y, x_1, y_1, 1, 1, text, None], [x, y, x_1, y_1]

    def _process_data(self, data):

        for key, _data in data.items():
            
            file_name = _data["image_file"]
            # image = Image.open(file_name).convert('L')

            image = Image.open(file_name)
            
            width, height = image.size

            labels = []
            vertexes = []
            for text, bbox_for_train, key_type, formal_key in zip(
                _data["texts"], _data["bboxes"], _data["keys_type"], _data["formal_keys"]):

                # construct the

                label = []
                if formal_key in self.key_index:
                    label.append(self.key_index[formal_key])
                    if key_type == 'key':
                        label.append(self.key_index[formal_key] * 2 - 1)
                    elif key_type == 'value':
                        label.append(self.key_index[formal_key] * 2)
                    else:
                        label.append(0)

                else:
                    label.append(0)
                    label.append(0)

                labels.append(label)

                # resize the bounding box
                bbox_for_train[0] = bbox_for_train[0] / width
                bbox_for_train[1] = bbox_for_train[1] / height
                bbox_for_train[2] = bbox_for_train[2] / width
                bbox_for_train[3] = bbox_for_train[3] / height

                bag_of_words = to_bag_of_words(text, 0, self.tok_to_id, self.blank_idx)
                vertexes.append(np.hstack((bbox_for_train, bag_of_words)))

            # construct the graph
            input_bboxs = _data["input_bboxes"]
            graph = Graph(input_bboxs)
            N = len(input_bboxs)
            A = graph.adj[:N, :, :N]

            _data["A"] = A
            _data["labels"] = np.array(labels)
            _data["image"] = 0
            _data["image_size"] = (width, height)
            _data["vertexes"] = np.array(vertexes)

    def _read_data(self):
        # read corpus, classes, key_types and data

        labels = load_samples(self.labels_dir)

        data = defaultdict(dict)

        all_formal_key = []

        count_formal = 0

        categories = []

        for key, label in labels.items():
            file_name = os.path.join(self.samples_dir, label["file_name"])
            # LOGGER.info(file_name)
            texts = []
            bboxes_for_train = []
            input_bboxes = []

            keys_type = []
            formal_keys = []

            patient_status = get_classification_label(label)

            for info in label["attributes"]["_via_img_metadata"]["regions"]:
                region_attributes = info["region_attributes"]
                shape_attributes = info["shape_attributes"]

                text = str(region_attributes["label"])
                text = text.replace('\n', '').replace(' ', '')
                texts.append(text)

                # construct the

                keys_type.append(region_attributes['key_type'])

                formal_key = LABEL_DICT.get(region_attributes['formal_key'], '')

                if len(formal_key) > 0:
                    # exclude ""
                    formal_key = formal_key.replace('\n', '').replace('\t', '')
                    all_formal_key.append(formal_key)
                else:
                    count_formal += 1
            
                formal_keys.append(formal_key)

                # construct the graph
                input_bbox_format, bbox_for_train = self.transform_graph_input(shape_attributes, text)

                bboxes_for_train.append(bbox_for_train)
                input_bboxes.append(input_bbox_format)

            category = self.class_index[patient_status]
            categories.append(category)

            data[key]["keys_type"] = keys_type
            data[key]["formal_keys"] = formal_keys
            data[key]["image_file"] = file_name
            data[key]["category"] = category
            # data[key]["text"] = "".join(texts)
            data[key]["texts"] = texts            
            data[key]["bboxes"] = bboxes_for_train
            data[key]["input_bboxes"] = input_bboxes

        # if data[key]["category"] == None:
        #     LOGGER.info(data[key]["category"])
        #     exit()

            # break

        count_time = Counter(all_formal_key)

        LOGGER.info(count_time)

        ## ratio for focal loss
        ratio_focal_formal = []
        ratio_focal_kv = []
        ## background ratio
        background_ratio = 1 / (count_formal /
                                count_time.most_common()[0][1])

        ratio_focal_formal.append(background_ratio)
        ratio_focal_kv.append(background_ratio)

        ratio_focal_formal.append(1.0)
        ratio_focal_kv.append(1.0)
        ratio_focal_kv.append(1.0)
        most_common = count_time.most_common()[0][1]

        all_formal_key = [count_time.most_common()[0][0]]
        
        for w in count_time.most_common(100)[1:]:
            ratio_focal_formal.append(most_common / w[1])
            ratio_focal_kv.append(most_common / w[1])
            ratio_focal_kv.append(most_common / w[1])

            all_formal_key.append(w[0])


        ratio_focal_category = [0 for _ in range(len(self.class_index))]

        count_time = Counter(categories)

        LOGGER.info(count_time)

        for i, w in enumerate(count_time.most_common(100)):
            if i == 0:
                # maximum class
                ratio_focal_category[w[0]] = 1
                most_common = w[1]
            elif w[0] == 0:
                # background class
                ratio_focal_category[w[0]] = most_common / w[1] * 1.2
            else:
                ratio_focal_category[w[0]] = most_common / w[1]


        key_index = {k: i + 1 for i, k in enumerate(list(all_formal_key))}
        # key_index[""] = 0
        
        return data, key_index, ratio_focal_kv, ratio_focal_formal, ratio_focal_category

    def get_train_loader(self):
        # REQUIRED

        train_list = load_path_list(self.train_split)

        # append ocr_extend data

        if self.extend_dataset != "":
            extend_train_list = ["ocr_extend_" + name for name in train_list]
            train_list.extend(extend_train_list)

        # remove some keys not in the data
        train_list = [key for key in train_list if key in self.data]


        image_transform = transforms.Compose(
            [
            transforms.RandomAffine(20, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(5, 5), 
                                    fillcolor=0),
            transforms.ToTensor()
            ]
        )

        text_transform = transforms.Compose(
            [
                TextNoise(self.tok_to_id, 0.3),
                # TextToTensor(100)
            ]
        )

        box_transform = BoxNoise(0.1)

        dataset = TransformSubset(self.data, train_list, image_transform, text_transform=None, box_transform=None)

        LOGGER.info(len(train_list))

        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
                          drop_last=True, pin_memory=True)

    def get_val_loader(self):
        # OPTIONAL
        val_list = load_path_list(self.val_split)

        if self.extend_dataset != "":
            extend_train_list = ["ocr_extend_" + name for name in val_list]
            val_list.extend(extend_train_list)

        # remove some keys not in the data
        val_list = [key for key in val_list if key in self.data]

        image_transform = transforms.Compose(
            [
            transforms.ToTensor()
            ]
        )

        text_transform = transforms.Compose(
            [
                TextNoise(self.tok_to_id, 0.00),
                # TextToTensor(100)
            ]
        )

        dataset = TransformSubset(self.data, val_list, image_transform)

        LOGGER.info(len(val_list))

        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                          drop_last=False, pin_memory=True)

    def get_test_loader(self):
        # OPTIONAL
        pass
