import os
from multiprocessing import Pool
import multiprocessing as mp
from itertools import product

# videos are the folder that contains the videos
# subfolder: valid, train, test

VAL_VID_PATH = os.path.join("../../videos/","valid" )
TRAIN_VID_PATH = os.path.join("../../videos/", "train")
TEST_VID_PATH = os.path.join("../../videos/", "test")

# Make folder for saving frames
output_folder_video_image = os.path.join("../../", "per_second_frames")

if not os.path.isdir(output_folder_video_image):
	os.mkdir(output_folder_video_image)

for type in ['train', 'test', 'valid']:
	if not os.path.isdir(os.path.join(output_folder_video_image, type)):
		os.mkdir(os.path.join(output_folder_video_image, type))

def extract_all_frames(vid_path, type, vid_dir):

	frame_save_path = os.path.join(output_folder_video_image, type, vid_path[:-len('.mp4')] if vid_path.endswith('.mp4') else None)

	if not os.path.isdir(frame_save_path):
		os.mkdir(frame_save_path)

	#Extract per-second-frame for each video.
	#https://stackoverflow.com/questions/33341303/ffmpeg-resource-temporarily-unavailable

	#Computing node requires single-threaded code
	cmd = ['ffmpeg', '-threads 1', '-hide_banner', '-i', os.path.join(vid_dir, vid_path), '-vf', 'fps=1', os.path.join(frame_save_path, '%d.jpg')]

	cmd = " ".join(cmd)
	os.system(cmd)

if __name__ == '__main__':
	train_vid_list = os.listdir(TRAIN_VID_PATH)
	val_vid_list = os.listdir(VAL_VID_PATH)
	test_vid_list = os.listdir(TEST_VID_PATH)

	#Using multi-processing
	
	pool = Pool(processes=mp.cpu_count())
	pool.starmap(extract_all_frames, product(train_vid_list, ['train'], [TRAIN_VID_PATH]))
	pool.terminate()

	pool = Pool(processes=mp.cpu_count())
	pool.starmap(extract_all_frames, product(val_vid_list, ['valid'], [VAL_VID_PATH]))
	pool.terminate()

	pool = Pool(processes=mp.cpu_count())
	pool.starmap(extract_all_frames, product(test_vid_list, ['test'], [TEST_VID_PATH]))
	pool.terminate()	