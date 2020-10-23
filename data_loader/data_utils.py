import os
import json
from collections import OrderedDict, Counter

import numpy as np
import glob
import pandas as pd
import pickle
import cv2
from PIL import Image

def load_json(path):
    with open(path, 'r', encoding='utf8') as f:
	    return json.loads(f.read())

def load_msau_corpus(corpus_path):
    corpus = load_json(corpus_path)
    corpus = ' ' + '$' + corpus # add ' ' and $ to the corpus, but I don't know why
    corpus = "".join(OrderedDict.fromkeys(corpus))  # eliminating repeated chars
    return corpus

def load_msau_classes(classes_path):
    classes = load_json(classes_path)

    if 'None' not in classes:
        classes = ['None'] + classes

    msau_classes = []
    for cls_ in classes:
        if cls_ != 'None':
            msau_classes.append('k_' + cls_)
            msau_classes.append('v_' + cls_)
        else:
            msau_classes.append(cls_)

    return msau_classes

def load_path_list(lst_path):
	res = []
	with open(lst_path, 'r', encoding='utf8') as f:
		for line in f.readlines():
			if len(line) > 0:
				line = line.strip('\n\t\r ')
				if len(line) > 0:
					res.append(line)	
	return res

def load_samples(samples_path):
    samples = {}
    for file_ in os.listdir(samples_path):
        file_path = os.path.join(samples_path, file_)
        samples[file_] = load_json(file_path)
    return samples

class RandomCrop:
    def __init__(self):
        pass
    
    def __call__(self):
        pass

if __name__ == "__main__":
    preprocess_engine = Kv_preprocessing_engine()
    ## prepare dictionary/formal key classes for dataset, take sony as example
    preprocess_engine.process_preprocessing('../../data')