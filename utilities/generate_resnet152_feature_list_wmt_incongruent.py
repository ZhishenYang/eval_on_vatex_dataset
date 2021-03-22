import os
import random

DATA_PATH = "../../videos/"
OUT_PATH = "/home/0/18M38305/Lab_space_yang/Video_guided_machine_translation/eval_on_vatex_dataset/data/resnet152/per_second_frames/incongruent/"
VIDEO_ID = "/home/0/18M38305/Lab_space_yang/Video_guided_machine_translation/eval_on_vatex_dataset/data/bpe/tok/"
SAVE_PATH = "/home/0/18M38305/Lab_space_yang/Video_guided_machine_translation/video_features/resnet152/per_second_frames"

train_list = os.listdir(os.path.join(DATA_PATH,"train"))
val_list = os.listdir(os.path.join(DATA_PATH,"valid"))
test_list = os.listdir(os.path.join(DATA_PATH,"test"))

output_file = open(os.path.join(OUT_PATH,"resnet152.incongruent.per_second_frame_train.id"), 'w')
f = open(os.path.join(VIDEO_ID, "train.id"))
output_list = []

for line in f:
	v_id = line.split("&")[0]
	path = os.path.join(SAVE_PATH, "train", v_id + "_train_resnet152_avgpool.npy")
	output_list.append(path)

random_list = random.sample(output_list, len(output_list))
for path in random_list:
	output_file.write(path+"\n")
output_file.close()

output_file = open(os.path.join(OUT_PATH,"resnet152.incongruent.per_second_frame_val.id"), 'w')
f = open(os.path.join(VIDEO_ID, "val.id"))
output_list = []

for line in f:
	v_id = line.split("&")[0]
	path = os.path.join(SAVE_PATH, "valid", v_id + "_valid_resnet152_avgpool.npy")
	output_list.append(path)

random_list = random.sample(output_list, len(output_list))
for path in random_list:
	output_file.write(path + "\n")
output_file.close()

output_file = open(os.path.join(OUT_PATH,"resnet152.incongruent.per_second_frame_public_test.id"), 'w')
f = open(os.path.join(VIDEO_ID, "public_test.id"))
output_list = []

for line in f:
	v_id = line.split("&")[0]
	path = os.path.join(SAVE_PATH, "test", v_id + "_test_resnet152_avgpool.npy")
	output_list.append(path)

random_list = random.sample(output_list, len(output_list))
for path in random_list:
	output_file.write(path + "\n")
output_file.close()

