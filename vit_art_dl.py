import os
import sys
# import numpy as np
import tensorflow as tf
sys.path.append('/home/shashank/tapestry/models')
from official.vision.serving import export_saved_model_lib
import tensorflow_models as tfm
import yaml
import warnings
warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tf.get_logger().setLevel('ERROR')

if __name__=="__main__": 

    exp_config = tfm.core.exp_factory.get_exp_config('vit_imagenet_pretrain')

    with open("/home/shashank/cv/vit_resnet_finetuning/configs/vit_config.yaml", "r") as file:
        override_params = yaml.full_load(file)
    exp_config.override(override_params, is_strict=False)

    logical_device_names = [logical_device.name for logical_device in tf.config.list_logical_devices()]
    
    if exp_config.runtime.mixed_precision_dtype == tf.float16:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')

    if exp_config.device == "GPU":   
        print("Using GPU")
        distribution_strategy = tf.distribute.MirroredStrategy()

    elif exp_config.device == "CPU" and exp_config.debug_mode :
        print("Using CPU")
        distribution_strategy = tf.distribute.OneDeviceStrategy(logical_device_names[0])


    model_dir = "/home/shashank/cv/vit_resnet_finetuning/test_run"

    with distribution_strategy.scope():
        os.mkdir(model_dir)
        task = tfm.core.task_factory.get_task(exp_config.task, logging_dir=model_dir) 

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