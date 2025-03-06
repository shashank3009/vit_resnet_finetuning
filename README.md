
# Fine-Tuning ResNets and Vision Transformers on Skin Cancer and Art DL Datasets


## Table of Contents
1. [Introduction](#introduction)
2. [Data Preparation](#data-preparation)
3. [Model Checkpoints](#model-checkpoints)
4. [Training](#training)
5. [Results](#results)
6. [References](#references)

---

## Introduction

This repository contains code for fine-tuning **ResNets** and **Vision Transformers (ViT)** on two datasets:
1. **Skin Cancer Dataset**: [marmal88/skin_cancer](https://huggingface.co/datasets/marmal88/skin_cancer)
2. **Art DL Dataset**: [ArtDL](https://artdl.org/)

The models are fine-tuned separately for each dataset.

It is based on the official [TensorFlow Model Garden](https://github.com/tensorflow/models). The code has been modified to use **focal cross-entropy loss** to better address the class imbalance in the datasets.

---

## Data Preparation

### Dataset Links
- **Skin Cancer Dataset**: [marmal88/skin_cancer](https://huggingface.co/datasets/marmal88/skin_cancer)
- **Art DL Dataset**: [ArtDL](https://artdl.org/)

### TFRecord Creation
The datasets are converted into **TFRecords** for efficient training. The class distribution is stratified across train, test, and validation splits.

To create TFRecords, I modified the script (convert_images_to_tfr.py) to suit for image classification based on the official TensorFlow script: [create_coco_tf_record.py](https://github.com/tensorflow/models/blob/master/official/vision/data/create_coco_tf_record.py).

We need to create a JSON file which has metadata about each image which is required to create tfrecords. I have shared an example file, and the command below to create tfrecords.

#### Example Command
```bash
python3 convert_images_to_tfr.py --logtostderr \
      --image_dir="/home/shashank/tensorflow_datasets/downloads/art_dl/ArtDL/JPEGImages" \
      --image_info_file="/home/shashank/tensorflow_datasets/downloads/art_dl/ArtDL/train_images_info.json" \
      --output_file_prefix="/home/shashank/tensorflow_datasets/tfrecords/ArtDL/train" \
      --num_shards=1
```

- **`image_dir`**: Directory containing the images.
- **`image_info_file`**: JSON file with metadata about each image.
- **`output_file_prefix`**: Path prefix for the output TFRecord files.
- **`num_shards`**: Number of shards to split the dataset into.

---

## Model Checkpoints

I used the following pre-trained checkpoints for ResNet and Vision Transformer:

- **ResNet-50 Checkpoint**: [resnet-50-i224.tar.gz](https://storage.googleapis.com/tf_model_garden/vision/resnet/resnet-50-i224.tar.gz)
- **Vision Transformer (ViT) Checkpoint**: [vit-deit-imagenet-b16.tar.gz](https://storage.googleapis.com/tf_model_garden/vision/vit/vit-deit-imagenet-b16.tar.gz)

After downloading, extract the checkpoints and update the paths in the training files. 

---

## Training

### Training Files
In the training file, it imports the pre-trained configurations, so for fine-tuning we need to modify some parameters based on our task.

There are four separate Python files for training:
1. **ResNet on Skin Cancer Dataset**: `resnet_skin_cancer.py`
2. **ResNet on Art DL Dataset**: `resnet_art_dl.py`
3. **Vision Transformer on Skin Cancer Dataset**: `vit_skin_cancer.py`
4. **Vision Transformer on Art DL Dataset**: `vit_art_dl.py`

### Modifications
- **Focal Loss**: The code has been modified to use **focal loss** to handle class imbalance.
- **GPU Training**: Training is performed on GPU for faster convergence. 

---

## Results

*Results will be updated soon.*

---

## References
1. TensorFlow Models Repository: [https://github.com/tensorflow/models](https://github.com/tensorflow/models)
2. Skin Cancer Dataset: [marmal88/skin_cancer](https://huggingface.co/datasets/marmal88/skin_cancer)
3. Art DL Dataset: [ArtDL](https://artdl.org/)
4. ResNet Checkpoint: [resnet-50-i224.tar.gz](https://storage.googleapis.com/tf_model_garden/vision/resnet/resnet-50-i224.tar.gz)
5. Vision Transformer Checkpoint: [vit-deit-imagenet-b16.tar.gz](https://storage.googleapis.com/tf_model_garden/vision/vit/vit-deit-imagenet-b16.tar.gz)
6. Image Classification Guide: [TensorFlow Models Vision](https://www.tensorflow.org/tfmodels/vision/image_classification)
