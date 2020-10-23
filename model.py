import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import pickle
import logging
import math
import json
import cv2
import re

from copy import deepcopy
from models.gcn import Attention_GCN_backbone, Jeff_Attention_GCN_backbone, SelfAttention_GCN_backbone, \
    SelfAttention_GCN_classifier, MultiAttention_GCN_backbone
from utils.graph import Graph
from PIL import Image

logger = logging.getLogger(__name__)

logger.setLevel(logging.DEBUG)

logger.addHandler(logging.StreamHandler())

class ClassificationModel():
    def __init__(self,
                 weights_path,
                 key_index,
                 tok_to_id,
                 class_index,
                 testing_mode=True,
                 device="-1",
                 model_backbone='Attention_GCN_backbone',
                 number_adj=4,
                 num_classes=3):

        self.model_backbone = model_backbone

        if testing_mode:
            assert isinstance(key_index, str) and isinstance(tok_to_id, str)

            try:
                with open(key_index, 'rb') as file:
                    self.key_index = pickle.load(file)
            except:
                with open(key_index, 'r') as file:
                    self.key_index = json.load(file)
            try:
                with open(tok_to_id, 'rb') as file:
                    self.tok_to_id = pickle.load(file)
            except:
                with open(tok_to_id, 'r') as file:
                    self.tok_to_id = json.load(file)

            try:
                with open(class_index, 'rb') as file:
                    self.class_index = pickle.load(file)
            except:
                with open(class_index, 'r') as file:
                    self.class_index = json.load(file)

        else:
            self.key_index = key_index
            self.tok_to_id = tok_to_id
            self.class_index = class_index

        self.index_key = {value: key for key, value in self.key_index.items()}

        self.index_class = {value: key for key, value in self.class_index.items()}

        # for key in list(self.key_index.keys()):
        #     self.index_key[self.key_index[key]] = key


        self.all_character_in_dic = list(self.tok_to_id.keys())

        backbone = eval(model_backbone)
        print(backbone)
        self.model = backbone(
            len(self.all_character_in_dic) + 4, number_adj,
            len(self.key_index) + 1, 
            self.key_index,
            num_classification=num_classes)

        # elif model_backbone == 'GCN':
        #     self.model = GCN(
        #         len(self.all_character_in_dic) + 4, number_adj,
        #         len(self.key_index) + 1)
        # elif model_backbone == 'Spatial_Attention_GCN':
        #     self.model = Spatial_Attention_GCN(
        #         len(self.all_character_in_dic) + 4, number_adj,
        #         len(self.key_index) + 1)
        # else:
        #     raise ImportError('No avaliable backbone')

        if device != '-1' and torch.cuda.is_available():
            device_list = device.split(",")
            self.device = f"cuda:{device_list[0]}"
            self.device_number = device
        else:
            self.device = 'cpu'

        try:
            self.model.load_state_dict(torch.load(weights_path, map_location="cpu"))
            # self.model.eval()
            self.model.to(self.device)
            logger.info('[classification] loading successfully....')
        except:
            logger.info(
                '[classification] Can not load the weight, please input the proper weight....')

        # torch.set_num_threads(24)
        torch.set_flush_denormal(True)

        self.eval()

    def to(self, device):
        self.model.to(device)

    def eval(self):
        self.model.eval()

    def train(self):
        self.model.train()

    def inference_preprocessing(self,
                                kv_output,
                                image_path=None,
                                debug=False):
        if type(image_path) is str:
            if os.path.exists(image_path):
                input_image = Image.open(image_path).convert('L')
            else:
                raise ValueError("No input for pre-processing")
        else:
            if type(image_path) is np.ndarray:
                input_image = Image.fromarray(image_path).convert('L')
            else:
                input_image = image_path.convert('L')

        height = input_image.size[1]
        width = input_image.size[0]

        box_for_graph = []
        feature_for_input = []
        for item in kv_output:
            if item["location"] is None:
                continue
            x1, y1, x2, y2 = item['location'][0][0], item['location'][0][
                1], item['location'][2][0], item['location'][2][1]
            text = item['text']
            box_for_graph.append([x1, y1, x2, y2, 1, 1, text, None])
            x1 = x1 / width
            x2 = x2 / width
            y1 = y1 / height
            y2 = y2 / height

            bag_of_word = np.zeros(len(self.all_character_in_dic))
            for character in str(text):
                if character in self.all_character_in_dic:
                    bag_of_word[self.tok_to_id[character]] += 1

            feature_single = []
            feature_single.append(x1)
            feature_single.append(y1)
            feature_single.append(x2)
            feature_single.append(y2)

            for index in bag_of_word:
                feature_single.append(index)

            feature_for_input.append(feature_single)

        box_for_graph = np.array(box_for_graph)
        feature_for_input = np.array(feature_for_input)

        ouput = Graph(box_for_graph)
        self.graph_output = ouput
        N = len(box_for_graph)
        A_adj = ouput.adj[:N, :4, :N]

        self.input_A = torch.tensor(A_adj).type(
            torch.FloatTensor).unsqueeze(0)
        self.input_V = torch.tensor(feature_for_input).type(
            torch.FloatTensor).unsqueeze(0)

        self.input_A, self.input_V = \
            self.input_A.to(self.device), self.input_V.to(self.device)

        image = np.array(input_image).astype(np.uint8)
        image = cv2.resize(image, (768, 768), interpolation=cv2.INTER_AREA)
        image = (image / 127.5) - 1

        self.input_img = torch.tensor(image).type(
            torch.FloatTensor).unsqueeze(0).unsqueeze(0)
        self.input_img = self.input_img.to(self.device)

        if debug:
            ouput._draw_debug_image()
            logger.info('Adj shape: {}'.format(self.input_A.shape))
            logger.info('V shape: {}'.format(self.input_V.shape))
            logger.info('Img shape: {}'.format(self.input_img.shape))

    @torch.no_grad()
    def process(self,
                kv_output,
                image_path,
                debug=False,
                filter_low_confidence=True):
        self.inference_preprocessing(kv_output, image_path, debug)
        _, output_final, output_classification = self.model(self.input_img, self.input_V, self.input_A)

        conf_classification, pred_classification = F.softmax(output_classification[0], -1).max(-1)
        conf_final, pred_final = F.softmax(output_final[0], -1).max(-1)
        conf_classification, pred_classification = \
            conf_classification.cpu().item(), pred_classification.cpu().item()

        p_text = ""
        pos = torch.where(pred_final == self.key_index["patient_status"] * 2)[0]
        if len(pos) != 0:
            for i in pos:
                if all(x in kv_output[i]["text"] for x in ["外来", "入院"]):
                    continue
                p_text += kv_output[i]["text"]

        d_text = ""
        pos = torch.where(pred_final == self.key_index["date"] * 2)[0]
        if len(pos) != 0:
            for i in pos:
                d_text += kv_output[i]["text"]

        output = {}
        output['location'] = [[0, 0] for _ in range(4)]

        output['text'] = ""
        output['patient_status'] = p_text
        output['date'] = d_text
        output['key_type'] = 'classification'
        output['formal_key'] = [self.index_class[pred_classification]]
        output['confidence'] = conf_classification

        # kv_output.append(output)

        return output

    def debug(self,
              image_input,
              font_='spatial_attention_graph/utils/NotoSansCJK-Black.ttc'):

        import matplotlib.pyplot as plt
        import matplotlib
        from matplotlib.font_manager import FontProperties

        myfont = FontProperties(fname=font_)

        image = Image.open(image_input).convert('L')
        image = np.array(image).astype(np.uint8)

        image = np.concatenate(
            (image.reshape(image.shape[0], image.shape[1], 1),
             image.reshape(image.shape[0], image.shape[1], 1),
             image.reshape(image.shape[0], image.shape[1], 1)),
            2)  # plot the image for matplotlib

        image_origin = image.copy()

        for item in self.output:
            x1, y1, x2, y2 = item['location'][0][0], item['location'][0][
                1], item['location'][2][0], item['location'][2][1]
            key_type = item['key_type']

            if key_type == 'key':
                image[y1:y2, x1:x2] = [255, 0, 0]
            elif key_type == 'value':
                image[y1:y2, x1:x2] = [0, 255, 0]

        image_origin = cv2.addWeighted(image, 0.6, image_origin, 0.4, 0)

        plt.figure(figsize=(20, 20))
        colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
        plt.imshow(image_origin)  # plot the image for matplotlib
        plt.rcParams.update({'font.size': 20})
        currentAxis = plt.gca()
        plt.axis('off')
        font = {'size': 10}
        matplotlib.rc('font', **font)
        start = time.time()

        for flow_output in self.output:
            coordination = flow_output['location']
            x0 = coordination[0][0]
            y0 = coordination[0][1]
            x1 = coordination[2][0]
            y1 = coordination[2][1]
            coords = (x0, y0), x1 - x0 + 1, y1 - y0 + 1
            color = colors[0]
            currentAxis.add_patch(
                plt.Rectangle(*coords,
                              fill=False,
                              edgecolor=color,
                              linewidth=2))
            if flow_output['formal_key'][0] != 'other':
                currentAxis.annotate(
                    flow_output['formal_key'][0] +
                    ': {}%'.format(np.round(flow_output['confidence'], 2)),
                    xy=(x0, y0),
                    fontproperties=myfont,
                    fontsize=10,
                    color='blue')