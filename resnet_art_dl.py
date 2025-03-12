import os
import sys
import numpy as np
import tensorflow as tf
sys.path.append('/home/shashank/tapestry/models')
from official.vision.serving import export_saved_model_lib
import tensorflow_models as tfm
import yaml
import pprint
import warnings
warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Reinitialize TF with the new log level
tf.get_logger().setLevel('ERROR')

if __name__=="__main__":    

    exp_config = tfm.core.exp_factory.get_exp_config('resnet_imagenet')

    with open("/home/shashank/cv/vit_resnet_finetuning/imagenet_resnet50_gpu.yaml", "r") as file:
        override_params = yaml.full_load(file)
    exp_config.override(override_params, is_strict=False)

    # Configure model
    exp_config.task.model.num_classes = 10
    exp_config.task.model.input_size = [224, 224, 3]

    # Configure training and testing data
    batch_size = 16

    exp_config.task.train_data.input_path = '/home/shashank/tensorflow_datasets/tfrecords/ArtDL/train-00000-of-00001.tfrecord'
    exp_config.task.train_data.global_batch_size = batch_size

    exp_config.task.validation_data.input_path = '/home/shashank/tensorflow_datasets/tfrecords/ArtDL/val-00000-of-00001.tfrecord'
    exp_config.task.validation_data.global_batch_size = batch_size

    exp_config.task.losses.alpha = 0.25
    exp_config.task.losses.gamma = 2

    exp_config.task.losses.use_focal_loss = False
    exp_config.task.losses.use_categorical_focal_loss = True


    exp_config.task.train_data.decode_jpeg_only = False
    exp_config.task.validation_data.decode_jpeg_only = False

    exp_config.task.train_data.image_field_key = 'image/encoded'
    exp_config.task.train_data.label_field_key = 'image/label'

    exp_config.task.validation_data.image_field_key = 'image/encoded'
    exp_config.task.validation_data.label_field_key = 'image/label'

    exp_config.task.init_checkpoint = '/home/shashank/cv/vit_resnet_finetuning/checkpoints/resnet-50-i224/'
    exp_config.task.init_checkpoint_modules = 'backbone'

    logical_device_names = [logical_device.name for logical_device in tf.config.list_logical_devices()]

    if exp_config.runtime.mixed_precision_dtype == tf.float16:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')

    if 'GPU' in ''.join(logical_device_names):   
        print("Using GPU")
        device = 'GPU'
        distribution_strategy = tf.distribute.MirroredStrategy()
    elif 'TPU' in ''.join(logical_device_names):
        print("Using TPU")
        device = 'TPU'
        tf.tpu.experimental.initialize_tpu_system()
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='/device:TPU_SYSTEM:0')
        distribution_strategy = tf.distribute.experimental.TPUStrategy(tpu)
    else:
        device = 'CPU'
        print("Using CPU")
        distribution_strategy = tf.distribute.OneDeviceStrategy(logical_device_names[0])
        
    debug_mode = False
    if debug_mode:
        device = 'CPU'
        print("Using CPU")
        distribution_strategy = tf.distribute.OneDeviceStrategy(logical_device_names[0])
        
    if device=='CPU':
        train_steps = 6
        exp_config.trainer.steps_per_loop = 2
    else:
        train_steps=90000
        exp_config.trainer.steps_per_loop = 1000

    exp_config.trainer.summary_interval = 1000
    exp_config.trainer.checkpoint_interval = train_steps
    exp_config.trainer.validation_interval = 9000
    exp_config.trainer.validation_steps =  -1
    exp_config.trainer.train_steps = train_steps
    exp_config.trainer.optimizer_config.learning_rate.type = 'cosine'
    exp_config.trainer.optimizer_config.learning_rate.cosine.decay_steps = train_steps
    exp_config.trainer.optimizer_config.learning_rate.cosine.initial_learning_rate = 0.0001
    exp_config.trainer.optimizer_config.warmup.linear.warmup_steps = 100

    print(train_steps, exp_config.trainer.validation_steps)

    model_dir = "/home/shashank/cv/vit_resnet_finetuning/model_artdl_resnet"

    with distribution_strategy.scope():
        os.mkdir(model_dir)
        task = tfm.core.task_factory.get_task(exp_config.task, logging_dir=model_dir) # ImageClassificationTask(exp_config.task, logging_dir=model_dir)

    model_training = True
    model_export = True

    if model_training:
        model, eval_logs = tfm.core.train_lib.run_experiment(
            distribution_strategy=distribution_strategy,
            task=task,
            mode='train_and_eval',
            params=exp_config,
            model_dir=model_dir,
            run_post_eval=True)      
        
    if model_export:
        export_saved_model_lib.export_inference_graph(
            input_type='image_tensor',
            batch_size=1,
            input_image_size=[224, 224],
            params=exp_config,
            checkpoint_path=tf.train.latest_checkpoint(model_dir),
            export_dir=model_dir+'/export')
        
    print("completed")