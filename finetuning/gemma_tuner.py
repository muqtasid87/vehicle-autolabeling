import logging
from unsloth import FastVisionModel, get_chat_template
from trl import SFTTrainer, SFTConfig
from unsloth.trainer import UnslothVisionDataCollator
from .tuner_base import BaseTuner, MetricsCallback

logger = logging.getLogger(__name__)

class GemmaTuner(BaseTuner):
    """Fine-tuner for the Gemma model."""

    def _get_default_hyperparameters(self):
        return {
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 4,
            "gradient_checkpointing": True,
            "gradient_checkpointing_kwargs": {"use_reentrant": False},
            "max_grad_norm": 0.3,
            "warmup_ratio": 0.03,
            "num_train_epochs": 3,
            "learning_rate": 2e-4,
            "logging_steps": 1,
            "save_strategy": "steps",
            "save_steps": 20,
            "optim": "adamw_torch_fused",
            "weight_decay": 0.01,
            "lr_scheduler_type": "cosine",
            "seed": 3407,
            "report_to": "none",
            "remove_unused_columns": False,
            "dataset_text_field": "",
            "dataset_kwargs": {"skip_prepare_dataset": True},
        }

    def _load_model_and_tokenizer(self):
        logger.info("Loading Gemma-3-4B-PT model and processor...")
        # Note: Gemma uses a 'processor' which is analogous to a tokenizer for Qwen
        self.model, self.tokenizer = FastVisionModel.from_pretrained(
            "unsloth/gemma-3-4b-it",
            load_in_4bit=True,
            use_gradient_checkpointing="unsloth",
        )
        
        # Apply the correct chat template for Gemma
        self.tokenizer = get_chat_template(self.tokenizer, "gemma-3")

    def _configure_peft(self):
        logger.info("Configuring PEFT model for Gemma...")
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
            target_modules="all-linear",
            modules_to_save=["lm_head", "embed_tokens"],
        )
    
    def _initialize_trainer(self, dataset):
        FastVisionModel.for_training(self.model)
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=dataset,
            # For Gemma, the processor's tokenizer is explicitly passed
            processing_class=self.tokenizer.tokenizer,
            data_collator=UnslothVisionDataCollator(self.model, self.tokenizer),
            args=SFTConfig(**self.hyperparameters),
            callbacks=[MetricsCallback(self.output_dir)]
        )
        return trainer