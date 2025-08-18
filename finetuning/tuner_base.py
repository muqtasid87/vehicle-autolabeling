import os
import torch
import logging
from abc import ABC, abstractmethod
from transformers import TrainerCallback
from trl import SFTConfig
from common.logging_setup import setup_logging_and_dir
from common.data_utils import load_and_process_data
import config

logger = logging.getLogger(__name__)

class MetricsCallback(TrainerCallback):
    """A custom callback to log metrics to a separate file."""
    def __init__(self, log_dir):
        self.metrics_logger = logging.getLogger("metrics")
        self.metrics_logger.propagate = False
        self.metrics_logger.handlers.clear()
        handler = logging.FileHandler(os.path.join(log_dir, 'metrics.log'))
        handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        self.metrics_logger.addHandler(handler)
        self.metrics_logger.setLevel(logging.INFO)

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if logs:
            metrics_str = f"Step {state.global_step}: " + ", ".join([f"{k}={v:.4f}" for k, v in logs.items()])
            self.metrics_logger.info(metrics_str)
            logger.info(metrics_str)


class BaseTuner(ABC):
    """Abstract base class for model fine-tuning."""
    def __init__(self, input_folder, images_subfolder, json_subfolder, hyperparameters):
        self.model_name = self.__class__.__name__.replace("Tuner", "")
        self.output_dir = setup_logging_and_dir("Finetuning", self.model_name)

        self.images_folder = os.path.join(input_folder, images_subfolder)
        self.json_folder = os.path.join(input_folder, json_subfolder)
        
        self.hyperparameters = self._get_default_hyperparameters()
        self.hyperparameters.update(hyperparameters) # Override defaults with user-provided args
        self.hyperparameters['output_dir'] = self.output_dir

        self.model = None
        self.tokenizer = None

    def _log_gpu_stats(self, stage="Initial"):
        if torch.cuda.is_available():
            gpu_stats = torch.cuda.get_device_properties(0)
            reserved_mem = round(torch.cuda.max_memory_reserved() / 1e9, 3)
            max_mem = round(gpu_stats.total_memory / 1e9, 3)
            logger.info(f"({stage}) GPU: {gpu_stats.name}, Max Memory: {max_mem} GB")
            logger.info(f"({stage}) Reserved Memory: {reserved_mem} GB")
            return reserved_mem
        return 0

    def convert_to_conversation(self, sample):
        """Converts a data sample to the conversation format for training."""
        return {
            "messages": [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": {config.SYSTEM_PROMPT}}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"{config.USER_PROMPT}"},
                        {"type": "image", "image": sample["image"]},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": sample["ground_truth"]}]
                },
            ]
        }

    def train(self):
        """Main training loop."""
        self._load_model_and_tokenizer()
        self._configure_peft()

        logger.info("Loading and processing dataset...")
        training_samples = load_and_process_data(self.images_folder, self.json_folder)
        converted_dataset = [self.convert_to_conversation(sample) for sample in training_samples]

        start_mem = self._log_gpu_stats(stage="Pre-training")

        trainer = self._initialize_trainer(converted_dataset)
        
        logger.info("Starting model training...")
        trainer_stats = trainer.train()
        
        end_mem = self._log_gpu_stats(stage="Post-training")
        
        logger.info(f"Training took {trainer_stats.metrics['train_runtime']:.2f} seconds.")
        logger.info(f"Peak memory usage for training: {end_mem - start_mem:.3f} GB.")

        final_model_dir = os.path.join(self.output_dir, "lora_model")
        self.model.save_pretrained(final_model_dir)
        self.tokenizer.save_pretrained(final_model_dir)
        logger.info(f"✅ Trained model adapters saved to {final_model_dir}")

    @abstractmethod
    def _get_default_hyperparameters(self):
        pass

    @abstractmethod
    def _load_model_and_tokenizer(self):
        pass
    
    @abstractmethod
    def _configure_peft(self):
        pass

    @abstractmethod
    def _initialize_trainer(self, dataset):
        pass