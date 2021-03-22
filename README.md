# Do Videos Guide Translations? Evaluation of a Video-Guided Machine Translation dataset 


This repository contains the implementation of our ViGIL2021 paper: [Do Videos Guide Translations? Evaluation of a Video-Guided Machine Translation dataset](https://vigilworkshop.github.io/static/papers-2021/29.pdf).

Abstract:
Video-guided machine translation (VMT) is a new multimodal machine translation task aimed at using videos to guide translation. Visual information gleaned from videos is expected to provide context in the translation progress. Results from the Video-guided Machine Translation Challenge 2020 suggest that multimodal models only have marginal performance improvements over their text-only counterparts. We hypothesize that this is caused by the simple and short video descriptions in VATEX, the dataset used in the challenge. In this study, we examine our hypothesis by conducting input-degradation, visual sensitivity experiments, and human evaluation of VATEX. The results indicate that textual descriptions of videos in VATEX are sufficient for translation, which prevents the visual context from videos to guide the translation

## Instruction

### Download dataset 
We will use VATEX v1.1 (Latest Version):

 ```
 mkdir data
 mkdir data/raw
 cd data/raw
```
We will download VATEX dataset into `data/raw` folder:

**Download training set:**

`wget  https://eric-xw.github.io/vatex-website/data/vatex_training_v1.0.json`

**Download validation set:**

`wget https://eric-xw.github.io/vatex-website/data/vatex_validation_v1.0.json`

**Download public-test set:**

`wget https://eric-xw.github.io/vatex-website/data/vatex_public_test_english_v1.1.json`

### Download videos 
Please save videos into  
1 `../videos/train`

2 `../videos/valid`

3 `../videos/test`

### Setup environment

1. `conda env create --name envname --file=environment_preprocess.yml`
2. `conda env create --name envname --file=environment_train.yml`
3. `./setup.sh`


### Download I3D features (provided by VATEX)

```bash
mkdir ./data/i3d/
cd ./data/i3d/
```
Then download official I3D Features from:

```
# Train & Validation Sets (3.0 GB): https://vatex-feats.s3.amazonaws.com/trainval.zip

# Public Test Set (634.9 MB): https://vatex-feats.s3.amazonaws.com/public_test.zip

wget  https://vatex-feats.s3.amazonaws.com/trainval.zip
wget  https://vatex-feats.s3.amazonaws.com/public_test.zip
  
```

Download I3D features into `./data/i3d/`, then extract features from `trainval.zip` into `train_val` folder,
and from `public_test.zip` into `public_test` folder. 


### ResNet-152 feature extraction 

1. First we need to extract per-second frames from videos.
`cd visual_feature_extraction` 
   
`python extract_1s_frame.py`

Extracted frames will be saved in `../../per_second_frames`   

2. Then let's extract visual features using ResNet-152 from extracted frames:

`mkdir -p ../../resnet152/per_second_frames/` 

```
CUDA_VISIBLE_DEVICES=0 python resnet_feature_extractor_wmt.py \
    -f ../../per_second_frames/train \ # Replace train with valid for validation set
    -b 32 \
    -n per_second_frames \
    -o resnet152 \
    -s ../../resnet152/per_second_frames/ &
```

### Generate feature path list
Go to folder `./utilities`

1. Generate i3d feature path list and incongruent path list:
```python
python generate_feature_list.py
python generate_incongrunent_list.py
```
2. Generate ResNet-152 feature path list and incongruent path list:
```python
python generate_resnet152_feature_list_wmt.py
python generate_resnet152_feature_list_wmt_incongruent.py
```
###Input Degradation
`cd scripts`
```
./color_deprivation.sh
./text_tokenization.sh
./verb_masking.sh
./noun_masking.sh
./progressive_masking.sh
```

### BPE
`cd scripts`
```
./make_bpe_vocab_progressive_masking.sh
./make_bpe_vocab.sh
./data-bpe_progressive_masking.sh -m 8000
./make_bpe.sh -m 8000
```

### Train models

There two subfolders in `./models` directory: `using_official_i3d` and `using_resnet_152`

In each subfolder:
1. `tok_bpe` is for complete dataset and incongruent visual feature experiments
2. input-degradation model configuration files are placed in `color_deprivation`  `noun_masking`  `verb_masking`

Training model follows the same procedures, the only different is that we need to copy feature path list for I3D for
models that use I3D and for ResNet for models that use ResNet.

Example of training a Vgt-shallow models using ResNet-152 features on complete dataset:

```
cp ../../../data/resnet152/per_second_frames/resnet152.per_second_frame_train.id .
cp ../../../data/resnet152/per_second_frames/resnet152.per_second_frame_val.id .
cp ../../../data/resnet152/per_second_frames/resnet152.per_second_frame_public_test.id .
```

Traing the model:
```
CUDA_VISIBLE_DEVICES=0 nmtpy train -C  vgt-shallow.conf
```

Checkpoints are saved in `ckpt/vgt-shallow` (defined in `vgt-shallow.conf`) directory.

Type `nmtpy train -h` for more details.

### Generate translation

1. Run following command to generate translation of `val` samples to `./output/vgt-shallow` using saved model `ckpt/vgt-shallow/12345.best.bleu.ckpt`:
```
CUDA_VISIBLE_DEVICES=0 nmtpy translate -b 512 -k 1 -u -m 40 -s val -o ./output/vgt-shallow   ./ckpt/vgt-shallow/12345.best.bleu.ckpt
```

Specify beam size by `-k`, batch size by `-b`, split by `-s`, and output directory by `-o`. Use `-u` to suppress `UNK` in translation.

Type `nmtpy translate -h` for more details.

2. For incongruenct visual feature test experiment:

First, we need to copy the incongruent feature path list to the current folder: 
`cp ../../../data/resnet152/per_second_frames/resnet152.incongruent.per_second_frame_val.id .`

Rename `resnet152.incongruent.per_second_frame_val.id` to  `resnet152.per_second_frame_val.id `

Run the generation command:
```
CUDA_VISIBLE_DEVICES=0 nmtpy translate -b 512 -k 1 -u -m 40 -s val -o ./output/vgt-shallow   ./ckpt/vgt-shallow/12345.best.bleu.ckpt
```
