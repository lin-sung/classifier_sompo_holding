import os
import random
import sys

data_folder = sys.argv[1]

##################### First Step
# first extract train and then execute the following codes

# train_files = os.listdir(os.path.join(data_folder, "labels"))

# with open(os.path.join(data_folder, "train.lst"), "w") as f:
#     written_thing = "\n".join(train_files)
#     print(written_thing)
#     f.write(written_thing)

###################### Second Step
## first extract valid and then execute the following codes

train_files = []
with open(os.path.join(data_folder, "train.lst"), "r") as f:
    read_files = f.readlines()

for file_ in read_files:
    train_files.append(file_.rstrip())

print(train_files)

total_files = os.listdir(os.path.join(data_folder, "labels"))

# train_files = random.sample(total_files, int(0.9 * len(total_files)))

# print(len(train_files), len(total_files))

valid_files = [file_ for file_ in total_files if file_ not in train_files]

# with open("data/train.lst", "w") as f:
#     written_thing = "\n".join(train_files)
#     print(written_thing)
#     f.write(written_thing)

with open(os.path.join(data_folder, "val.lst"), "w") as f:
    written_thing = "\n".join(valid_files)
    print(written_thing)
    f.write(written_thing)

print(len(valid_files), len(train_files), len(total_files))


