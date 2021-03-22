import argparse
import json
import multiprocessing as mp
import os
import sys
from multiprocessing import Pool

import spacy
from matplotlib.colors import is_color_like




def info(s, **kwargs):
	print(s, file=sys.stderr, **kwargs)
	print(s, file=sys.stdout, **kwargs)


def masking(lang, sentence, task, n=0):

	# Tokenization only, Chinese
	if lang == "zh":

		# Simple space-based tokenization gives a better peroformance than using Jieba that comes with Spacy.
		# Give option to use Jieba

		if (args.chinese_tokenizer == 'jieba'):
			nlp = zh_nlp
			doc = nlp(sentence)
			tokens = [token for token in doc]

			result = ""
			for token in doc:
				result = result + " " + token.text
			return result
		else:
			result = ""
			for token in sentence.strip():
				result = result + " " + token
			return result

	# Tokenization only, English
	if task == "tok" and lang == 'en':
		nlp = en_nlp
		doc = nlp(sentence)
		tokens = [token for token in doc]
		num_tokens = 0
		total_tokens = len(tokens)

		result = ""
		for token in doc:
			result = result + " " + token.text
			if (token.pos_ != "PUNCT"):
				num_tokens = num_tokens + 1

		return result.lower(), 0, num_tokens, total_tokens

	nlp = en_nlp
	masking_n = 0

	doc = nlp(sentence)
	result = ""

	tokens = [token for token in doc]
	num_tokens = 0
	total_tokens = len(tokens)

	if (task == 'color'):
		for token in doc:
			word = ""
			if (token.pos_ != "PUNCT"):
				num_tokens = num_tokens + 1

			if (is_color_like(token.text.lower()) and token.pos_ == 'ADJ'):
				word = "[c]"
				masking_n = masking_n + 1
			else:
				word = token.text.lower()

			if (len(result) == 0):
				result = result + word
			else:
				result = result + " " + word

	if (task == 'noun'):
		for token in doc:
			word = ""
			if (token.pos_ != "PUNCT"):
				num_tokens = num_tokens + 1

			if (token.pos_ == 'NOUN'):
				word = "[n]"
				masking_n = masking_n + 1
			else:
				word = token.text.lower()

			if (len(result) == 0):
				result = result + word
			else:
				result = result + " " + word

	if (task == 'verb'):
		for token in doc:
			word = ""

			if (token.pos_ != "PUNCT"):
				num_tokens = num_tokens + 1

			if (token.pos_ == "VERB"):
				word = "[v]"
				masking_n = masking_n + 1
			else:
				word = token.text.lower()

			if (len(result) == 0):
				result = result + word
			else:
				result = result + " " + word

	if (task == 'progress_masking'):

		num_tokens = 0
		count_i = 0

		for i in reversed(range(len(tokens))):

			word = tokens[i].text.lower()

			if (tokens[i].pos_ != "PUNCT"):
				num_tokens = num_tokens + 1

			if (count_i < int(n)):
				tokens[i] = '[p]'
				masking_n = masking_n + 1
			else:
				tokens[i] = word
			count_i = count_i + 1

		result = ' '.join([token for token in tokens])

	return result, masking_n, num_tokens, total_tokens


def processing(videos):
	video_ids = []
	en_caps = []
	zh_caps = []
	total_masked_tokens = 0
	total_en_tokens = 0
	total_token_num = 0
	max_en_tokens = 0
	max_total_tokens = 0

	for video in videos:
		video_ids.append(video['videoID'])

		for cap in video['enCap'][5:]:
			processed_en_sen, masked_tokens, num_tokens, total_tokens = masking('en', cap, args.task,
			                                                                    args.num_masking, )
			en_caps.append(processed_en_sen)

			total_masked_tokens = total_masked_tokens + masked_tokens
			total_en_tokens = total_en_tokens + num_tokens
			total_token_num = total_token_num + total_tokens

			if (num_tokens > max_en_tokens):
				max_en_tokens = num_tokens

			if (num_tokens > max_total_tokens):
				max_total_tokens = total_tokens

		if ('zh' in langs):
			for cap in video['chCap'][5:]:
				processed_cn_sen = masking('zh', cap, args.task, args.num_masking)
				zh_caps.append(processed_cn_sen)

	return video_ids, en_caps, zh_caps, total_masked_tokens, total_en_tokens, total_token_num, max_en_tokens, max_total_tokens


def load_obtained_video_list(split)->list:

	# we need to have a list of downloaded videos, will only train, valid, and test on those videos
	obtained_vid_list = []

	dir_path = "../../videos/" # folder contains videos
	with open(os.path.join(dir_path, "obtained_" + split + "_video_list")) as f:
		for line in f:
			obtained_vid_list.append(line.split('.mp4')[0])

	print(obtained_vid_list[0])
	print("Number of obtained videos: {}".format(len(obtained_vid_list)))
	return obtained_vid_list

def main(args):

	with open(args.json, 'r') as file:
		data = json.load(file)  # List, each item in the list is a dictionary

	# Key in each dictionary: dict_keys(['videoID', 'enCap', 'chCap'])
	# Let's go ahead and process the video
	data_set = []

	for video in data:
		data_set.append([video])

	num_videos = len(data_set)
	print(f"number of videos in {args.split} dataset: {num_videos}, number of downloaded {args.split} videos {len(obtained_vid_list)}")

	pool = Pool(processes=mp.cpu_count())
	results = pool.map(processing, data_set)
	pool.terminate()
	print(len(results))

	assert (len(results) == num_videos)

	video_ids_lists = []
	processed_caps = {}
	for lang in langs:
		processed_caps[lang] = []

	total_masked_tokens = 0
	total_en_tokens = 0
	total_token_num = 0
	final_max_en_tokens = 0
	final_max_total_tokens = 0

	for i, item in enumerate(results):
		if (str(item[0][0]) not in  obtained_vid_list):
			#print("Video (could not download): ",item[0][0] )
			continue

		video_ids_lists.append(item[0][0])
		for i, lang in enumerate(langs):
			processed_caps[lang].append(item[i + 1])

		total_masked_tokens = total_masked_tokens + int(item[-5])
		total_en_tokens = total_en_tokens + int(item[-4])
		total_token_num = total_token_num + int(item[-3])

		if (int(item[-2]) > final_max_en_tokens):
			final_max_en_tokens = int(item[-2])
		if (int(item[-1]) > final_max_total_tokens):
			final_max_total_tokens = int(item[-1])

	#info('Saving data...')

	for lang in langs:

		if (args.task == 'progress_masking'):
			output = open(args.out + '.{}.{}'.format(args.num_masking, lang), 'w')
		else:
			output = open(args.out + '.{}'.format(lang), 'w')

		for caps in processed_caps[lang]:
			for i, cap in enumerate(caps):
				output.writelines(cap + '\n')

	video_id = open(args.out + '.id', 'w')

	if (args.task == 'progress_masking'):
		video_id = open(args.out + '.{}.id'.format(args.num_masking), 'w')
	print("Number of video, after processing", len(video_ids_lists))

	for vid_id in video_ids_lists:
		for i in range(0, 5):
			video_id.writelines(["{}&{}\n".format(vid_id, i)])

	info('Task: {} \n, Total tokens: {}, Total none-punctuation tokens: {}, Total masked tokens: {}, Masked/Total: {}'.format(
		args.task, total_token_num, total_en_tokens, total_masked_tokens,
		float(total_masked_tokens) / float(total_token_num)))

	info('Maximum tokens without punctuation: {}, Maximum tokens with punctuation: {}'.format(final_max_en_tokens,
	                                                                                          final_max_total_tokens))


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-j', '--json', required=True)
	parser.add_argument('-o', '--out', required=True)
	parser.add_argument('-s', '--split', required=True)
	parser.add_argument('-l', '--language', nargs='*', choices=['en', 'zh'], default=['en', 'zh'])
	parser.add_argument('-b', '--chinese_tokenizer', default='space', choices=['jieba', 'space'])
	parser.add_argument('-t', '--task', required=True, choices=['color', 'noun', 'verb', 'tok', 'progress_masking'],
	                    default=['tok'])
	parser.add_argument('-n', '--num_masking', default=['0'])
	args = parser.parse_args()

	info(args)
	langs = args.language

	en_nlp = spacy.load("en_core_web_lg")
	zh_nlp = spacy.load("zh_core_web_lg")  # Load spacy's Chinese language model, Jieba will be used from tokenization
	obtained_vid_list = load_obtained_video_list(args.split)

	max_len = 0
	main(args)
