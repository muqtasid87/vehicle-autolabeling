import logging
from unsloth import FastVisionModel
from trl import SFTTrainer, SFTConfig
from unsloth.trainer import UnslothVisionDataCollator
from .tuner_base import BaseTuner, MetricsCallback

logger = logging.getLogger(__name__)

class QwenTuner(BaseTuner):
    """Fine-tuner for the Qwen2.5-VL model."""

    def _get_default_hyperparameters(self):
        return {
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 4,
            "gradient_checkpointing": True,
            "warmup_steps": 5,
            "num_train_epochs": 3,
            "learning_rate": 2e-4,
            "logging_steps": 1,
            "save_strategy": "steps",
            "save_steps": 20,
            "optim": "adamw_8bit",
            "weight_decay": 0.01,
            "lr_scheduler_type": "linear",
            "seed": 3407,
            "report_to": "none",
            "remove_unused_columns": False,
            "dataset_text_field": "",
        }

    def _load_model_and_tokenizer(self):
        logger.info("Loading Qwen2.5-VL-3B-Instruct model and tokenizer...")
        self.model, self.tokenizer = FastVisionModel.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct",
            load_in_4bit=True,
            use_gradient_checkpointing="unsloth",
        )

    def _configure_peft(self):
        logger.info("Configuring PEFT model for Qwen...")
        self.model = FastVisionModel.get_peft_model(
            self.model,
            finetune_vision_layers=True,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
            r=16,
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )

    def _initialize_trainer(self, dataset):
        FastVisionModel.for_training(self.model)
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            data_collator=UnslothVisionDataCollator(self.model, self.tokenizer),
            train_dataset=dataset,
            args=SFTConfig(**self.hyperparameters),
            callbacks=[MetricsCallback(self.output_dir)]
        )
        return trainer