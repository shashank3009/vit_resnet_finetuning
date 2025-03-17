## **Data Preparation**

---

### **Dataset Links**
- **Skin Cancer Dataset**: [marmal88/skin_cancer](https://huggingface.co/datasets/marmal88/skin_cancer)  
- **Art DL Dataset**: [ArtDL](https://artdl.org/)  

---

### **TFRecord Creation**
To enable efficient training, the datasets were converted into **TFRecords**. The class distribution was carefully stratified across **train**, **test**, and **validation** splits to ensure balanced representation.

The TFRecords were created by modifying the official TensorFlow script for object detection, [create_coco_tf_record.py](https://github.com/tensorflow/models/blob/master/official/vision/data/create_coco_tf_record.py), to suit the image classification task. The modified script, **`convert_images_to_tfr.py`**, was used for this purpose.

#### **Metadata JSON File**
A JSON file containing metadata about each image (e.g., file paths, labels, and other relevant information) is required to create TFRecords. An example [ipython notebook](./data_prepare/data_prepare.ipynb) has been shared to demonstrate how to generate this metadata JSON file.

#### **Example Command**
Below is an example command to create TFRecords using the modified script:

```bash
python3 convert_images_to_tfr.py --logtostderr \
      --image_dir="/home/shashank/tensorflow_datasets/downloads/skin_cancer" \
      --image_info_file="/home/shashank/tensorflow_datasets/downloads/skin_cancer/train_images_info.json" \
      --output_file_prefix="/home/shashank/tensorflow_datasets/tfrecords/skin_cancer/train" \
      --num_shards=1
```

- **`image_dir`**: Directory containing the images.  
- **`image_info_file`**: Path to the JSON file containing metadata about each image.  
- **`output_file_prefix`**: Prefix for the output TFRecord files.  
- **`num_shards`**: Number of shards to split the dataset into.  

---

This structured approach ensures efficient data handling and seamless integration with TensorFlow's training pipeline. Let me know if you need further refinements!