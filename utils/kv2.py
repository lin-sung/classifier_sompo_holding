import json
import glob
import copy
import os
from pathlib import Path

import re

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 7階部分\u300059.82
output_kv=[
  {
    "location": [
      [
        1505,
        2606
      ],
      [
        1672,
        2606
      ],
      [
        1672,
        2662
      ],
      [
        1505,
        2662
      ]
    ],
    "type": "total_area",
    "key_type": "value",
    "text": "7階部分",
    "confidence": 0.8831
  },
  {
    "location": [
      [
        1975,
        2606
      ],
      [
        2056,
        2606
      ],
      [
        2056,
        2659
      ],
      [
        1975,
        2659
      ]
    ],
    "type": "total_area",
    "key_type": "value",
    "text": "59",
    "confidence": 0.9897
  },
  {
    "location": [
      [
        1980,
        2607
      ],
      [
        2058,
        2607
      ],
      [
        2058,
        2660
      ],
      [
        1980,
        2660
      ]
    ],
    "type": "total_area",
    "key_type": "value",
    "text": "82",
    "confidence": 0.9897
  },
]

# 1階\u300049.81\n2階\u300050.22
output_kv2=  [{
    "location": [
      [
        820,
        554
      ],
      [
        866,
        554
      ],
      [
        866,
        581
      ],
      [
        820,
        581
      ]
    ],
    "type": "total_area",
    "key_type": "value",
    "text": "2階",
    "confidence_by_character": [
      0.9998089671134949,
      0.9999459981918335
    ],
    "confidence_by_field": 0.9998089671134949
  },
    {
    "location": [
      [
        820,
        524
      ],
      [
        866,
        524
      ],
      [
        866,
        552
      ],
      [
        820,
        552
      ]
    ],
    "type": "total_area",
    "key_type": "value",
    "text": "1階",
    "confidence_by_character": [
      0.9946974515914917,
      0.9999924898147583
    ],
    "confidence_by_field": 0.9946974515914917
  },
    {
    "location": [
      [
        955,
        520
      ],
      [
        1062,
        520
      ],
      [
        1062,
        549
      ],
      [
        955,
        549
      ]
    ],
    "type": "total_area",
    "key_type": "value",
    "text": "4981",
    "confidence_by_character": [
      0.9992103576660156,
      0.999408483505249,
      0.9993267059326172,
      0.9978658556938171
    ],
    "confidence_by_field": 0.9978658556938171
  },
  {
    "location": [
      [
        955,
        549
      ],
      [
        1064,
        549
      ],
      [
        1064,
        579
      ],
      [
        955,
        579
      ]
    ],
    "type": "total_area",
    "key_type": "value",
    "text": "5022",
    "confidence_by_character": [
      0.9997432827949524,
      0.998229444026947,
      0.9984353184700012,
      0.9989815354347229
    ],
    "confidence_by_field": 0.998229444026947
  },]


def form_cell():
    ret = confirm(collection=output_kv2)
    for r in ret:
        print(r)
    print("_______EXAMPLE_______")
    ret = h_process(collection=output_kv2, field_name="total_area", separator="__COL__", threshold=5)
    ret = v_process(collection=ret, field_name="total_area", separator="__ROW__", threshold=500)
    print(ret)

def form_json():
    files = glob.glob("debug_data-148/**/kv.json", recursive=True)
    # files = ["debug_data-148/12_22_1_PoC_House Registration certificate\kv.json"] # good case
    # files = ["debug_data-148/10_10_PoC_Mansion Registration Certificate\kv.json"] # good case
   
    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as fd:
            print("filenam: ", file_path)
            collection = json.load(fd)
            ret = confirm(collection=collection)
            for r in ret:
                print(r)
            print("_______EXAMPLE_______")
            ret = h_process(collection=collection)
            ret = v_process(collection=ret)
            _name = os.path.basename(file_path)
            path = Path(file_path)
            # print(path.parent)
            # print("_name", _name)
            newpath = "conv_json\\" + str(path.parent)
            if not os.path.exists(newpath):
                os.makedirs(newpath)
            save_json(f"{newpath}/{_name}", ret)
            print(ret)

# filenam:  debug_data-148/12_22_1_PoC_House Registration certificate\kv.json

def h_process(collection=[], field_name="total_area", threshold=5, separator=""):
    candidate = []
    for data in collection:
        key_type = data["key_type"]
        formal_key = data["type"]
        if formal_key == field_name and key_type == "value":
            candidate.append(data)

    cond = lambda obj: obj["location"][0][0]
    sorted_candidate = sorted(candidate, key=cond) 
    # print("sorted_candidate", sorted_candidate)

    final_candidate = []
    idx = 0
    while True:
        if len(sorted_candidate) == 0:
            break

        min_x = 1000000000000000000000000
        min_y = 1000000000000000000000000
        min_data = None
        for data in sorted_candidate:
            x1 = data["location"][0][0]
            y1 = data["location"][0][1]
            if y1 < min_y and x1 < min_x:
                min_y = y1
                min_x = x1
                min_data = data

        print("min_data", min_data)
        sorted_candidate.remove(min_data)
        final_candidate.append([])
        final_candidate[idx].append(min_data)

        _sorted_candidate = copy.deepcopy(sorted_candidate)
        for _data in _sorted_candidate:
            _min_y = _data["location"][0][1]
            if abs(_min_y - min_y) <= threshold:
                for data in sorted_candidate:
                  if data == _data:
                    final_candidate[idx].append(data)
                    sorted_candidate.remove(data)

        idx = idx + 1

    for arr in final_candidate:
        print("_______ prediction ____________")
        for obj in arr:
            print(obj)

    first_obj_ids = []
    for arr in final_candidate:
        first_obj_ids.append(id(arr[0]))
    print("first_obj_ids", first_obj_ids)

    all_ids = []
    for arr in final_candidate:
        for obj in arr:
            all_ids.append(id(obj))
    print("all_ids", all_ids)

    removed_ids = list(set(all_ids) - set(first_obj_ids))
    print("remove_ids", removed_ids)

    for data in collection:
        if id(data) in first_obj_ids:
            _idx = first_obj_ids.index(id(data))
            new_text_arr = []
            for obj in final_candidate[_idx]:
                text = str(obj["text"]).strip()
                text = text.replace("㎡", "")
                text = text.replace("m", "")
                text = text.replace("m²", "")
                text = text.replace("m³", "")
                text = text.replace(")", "")
                text = text.replace("(", "")
                text = text.replace("）", "")
                text = text.replace("（", "")
                text = text.replace(" ", "")
                text = text.replace("　", "")
                text = text.replace(":", ".")
                text = text.replace("：", ".")
                if "階" in text:
                    text = f"{text}　" # add zenkaku space
                new_text_arr.append(text)

            # check if text containes more than 3 letters at end
            new_text = separator.join(new_text_arr)
            def replace(x):
              txt = x.group()
              return txt[:len(txt)-2] + "." + txt[len(txt)-2:]
            data["text"] = re.sub(r'\d{3,}$', replace, new_text)

            x1, y1, x2, y2 = get_max_rect(final_candidate[_idx])
            data["location"][0][0] = x1
            data["location"][0][1] = y1
            data["location"][2][0] = x2
            data["location"][2][1] = y2

    print("len of  coll", len(collection))
    ret_collection = copy.deepcopy(collection)
    # print("len of  coll", len(collection))
    for idx, data in enumerate(collection):
        if id(data) in removed_ids:
            ret_collection.remove(data)

    print("len of  coll", len(ret_collection))
    return ret_collection 


def confirm(collection=[], field_name="total_area"):
    candidate = []
    for data in collection:
        key_type = data["key_type"]
        formal_key = data["type"]

        if formal_key == field_name: 
            candidate.append(data)
    return candidate


def v_process(collection=[], field_name="total_area", threshold=500, separator="\n"):
    candidate = []
    for data in collection:
        key_type = data["key_type"]
        formal_key = data["type"]

        if formal_key == field_name and key_type == "value":
            candidate.append(data)

    # sort candidate by Y value
    virtical_cond = lambda obj: obj["location"][0][1] # y1
    sorted_candidate = sorted(candidate, key=virtical_cond) 

    new_order={0:[],1:[],2:[],3:[],4:[],5:[]} 
    sort_correct_list=sorted_candidate
    order_=0
    
    while 1:
        if len(sort_correct_list)>0:
            count_=0
            for item in sort_correct_list:
                if count_==0:
                    new_order[order_].append(item)  
                else:
                    if abs(item['location'][0][1]-new_order[order_][0]['location'][0][1])<threshold:
                        new_order[order_].append(item)
                count_+=1
            for item in new_order[order_]:
                sort_correct_list.remove(item)
            order_+=1
        else:
            break
    # print(new_order)

    first_obj_ids = []
    for obj in new_order:
        if len(new_order[obj]) > 1:
            _id = id(new_order[obj][0])
            first_obj_ids.append(_id)
    print("first_obj_ids", first_obj_ids)

    all_ids = []
    for obj in new_order:
        if len(new_order[obj]) > 1:
            for data in new_order[obj]:
                _id = id(data)
                all_ids.append(_id)
    print(all_ids)

    removed_ids = list(set(all_ids) - set(first_obj_ids))
    print(removed_ids)

    for data in collection:
        if id(data) in first_obj_ids:
            idx2 = first_obj_ids.index(id(data))
            labels = []
            for arr in new_order[idx2]:
                text = arr["text"]
                labels.append(text)
            data["text"] = separator.join(labels)
            x1, y1, x2, y2 = get_max_rect(new_order[idx2])
            data["location"][0][0] = x1
            data["location"][0][1] = y1
            data["location"][2][0] = x2
            data["location"][2][1] = y2

    print("len of coll2:", len(collection))
    ret_collection = copy.deepcopy(collection)
    for data in collection:
        if id(data) in removed_ids:
            ret_collection.remove(data)

    print("len of coll2:", len(ret_collection))
    return ret_collection 

def save_json(path, ret):
    text = json.dumps(
            ret, 
            sort_keys=True, 
            indent=4, 
            separators=(',', ': '),
            ensure_ascii=False,
        )

    with open(path, "w", encoding='utf-8') as fd:
        fd.write(text)
        fd.close()

def get_max_rect(arr):
    x1 = 111111111110
    y1 = 111111111110
    x2 = 0
    y2 = 0
    for obj in arr:
        loc = obj["location"]
        _x1 = loc[0][0]
        _y1 = loc[0][1]
        _x2 = loc[2][0]
        _y2 = loc[2][1]
        if _x1 < x1:
            x1 = _x1
        if _y1 < y1:
            y1 = _y1
        if _x2 > x2:
            x2 = _x2
        if _y2 > y2:
            y2 = _y2
    return x1, y1, x2, y2

  

if __name__ == "__main__":
    # form_json()
    form_cell()
