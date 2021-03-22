import os

DATA_PATH = "../../videos/"
OUT_PATH = "/home/0/18M38305/Lab_space_yang/Video_guided_machine_translation/eval_on_vatex_dataset/data/i3d/"
VIDEO_ID = "/home/0/18M38305/Lab_space_yang/Video_guided_machine_translation/eval_on_vatex_dataset/data/bpe/tok/"

# List of downloaded videos
train_list = os.listdir(os.path.join(DATA_PATH, "train"))
val_list = os.listdir(os.path.join(DATA_PATH, "valid"))
test_list = os.listdir(os.path.join(DATA_PATH, "test"))

output_file = open(os.path.join(OUT_PATH, "i3d.train.id"), 'w')

f = open(os.path.join(VIDEO_ID, "train.id"))

for line in f:
    v_id = line.split("&")[0]
    path = os.path.join(OUT_PATH,"train_val", v_id +".npy")
    output_file.write(path+"\n")
output_file.close()

output_file = open(os.path.join(OUT_PATH,"i3d.val.id"), 'w')
f = open(os.path.join(VIDEO_ID, "val.id"))
for line in f:
    v_id = line.split("&")[0]
    path = os.path.join(OUT_PATH, "train_val", v_id  +".npy")
    output_file.write(path + "\n")
output_file.close()

output_file = open(os.path.join(OUT_PATH,"i3d.public_test.id"), 'w')
f = open(os.path.join(VIDEO_ID, "public_test.id"))
for line in f:
    v_id = line.split("&")[0]
    path = os.path.join(OUT_PATH, "public_test", v_id  +".npy")
    output_file.write(path + "\n")
output_file.close()