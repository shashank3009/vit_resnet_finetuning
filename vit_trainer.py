import os
import sys
import logging
import warnings
import yaml
import numpy as np
import tensorflow as tf
from official.vision.serving import export_saved_model_lib
import tensorflow_models as tfm

# Suppress warnings and TensorFlow logs
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add custom model path to sys.path
sys.path.append('/home/shashank/tapestry/models')

class ViTModelTrainer:
    def __init__(self, config_path, model_dir):
        self.config_path = config_path
        self.model_dir = model_dir
        self.exp_config = None
        self.distribution_strategy = None
        self.task = None

    def load_config(self):
        """Load and override experiment configuration from YAML file."""
        try:
            self.exp_config = tfm.core.exp_factory.get_exp_config('vit_imagenet_pretrain')
            with open(self.config_path, "r") as file:
                override_params = yaml.full_load(file)
            self.exp_config.override(override_params, is_strict=False)
            logger.info("Configuration loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise

    def setup_distribution_strategy(self):
        """Set up the distribution strategy based on the configuration."""
        logical_device_names = [logical_device.name for logical_device in tf.config.list_logical_devices()]

        if self.exp_config.runtime.mixed_precision_dtype == tf.float16:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')

        if self.exp_config.device == "GPU":
            logger.info("Using GPU")
            self.distribution_strategy = tf.distribute.MirroredStrategy()
        elif self.exp_config.device == "CPU" and self.exp_config.debug_mode:
            logger.info("Using CPU")
            self.distribution_strategy = tf.distribute.OneDeviceStrategy(logical_device_names[0])
        else:
            raise ValueError("Invalid device configuration.")

    def setup_task(self):
        """Set up the task for model training."""
        try:
            os.makedirs(self.model_dir)
            with self.distribution_strategy.scope():
                self.task = tfm.core.task_factory.get_task(self.exp_config.task, logging_dir=self.model_dir)
            logger.info("Task setup completed.")
        except Exception as e:
            logger.error(f"Either Model directory already exists or failed to set up task: {e}")
            raise

    def train_model(self):
        """Train the model and evaluate it."""
        try:
            logger.info("Starting model training...")
            model, eval_logs = tfm.core.train_lib.run_experiment(
                distribution_strategy=self.distribution_strategy,
                task=self.task,
                mode='train_and_eval',
                params=self.exp_config,
                model_dir=self.model_dir,
                run_post_eval=True
            )
            logger.info("Model training completed.")
            return model, eval_logs
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            raise

    def export_model(self, checkpoint_path):
        """Export the trained model for inference."""
        try:
            export_dir = os.path.join(self.model_dir, 'export')
            logger.info(f"Exporting model to {export_dir}...")
            export_saved_model_lib.export_inference_graph(
                input_type='image_tensor',
                batch_size=1,
                input_image_size=[224, 224],
                params=self.exp_config,
                checkpoint_path=checkpoint_path,
                export_dir=export_dir
            )
            logger.info("Model export completed.")
        except Exception as e:
            logger.error(f"Model export failed: {e}")
            raise


def main():
    # Configuration paths
    config_path = "/home/shashank/cv/vit_resnet_finetuning/vit_config.yaml"
    model_dir = "/home/shashank/cv/vit_resnet_finetuning/vit_art_dl_focal_default_bs16"

    # Initialize trainer
    trainer = ViTModelTrainer(config_path, model_dir)

    # Load configuration
    trainer.load_config()

    # Set up distribution strategy
    trainer.setup_distribution_strategy()

    # Set up task
    trainer.setup_task()

    # Train the model
    model_training = True
    if model_training:
        model, eval_logs = trainer.train_model()

    # Export the model
    model_export = True
    if model_export:
        checkpoint_path = tf.train.latest_checkpoint(model_dir)
        trainer.export_model(checkpoint_path)

    logger.info("Process completed.")


if __name__ == "__main__":
    main()