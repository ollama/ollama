"""Training configuration schema.

Defines the hyperparameters and hardware settings for fine-tuning jobs.
"""

from pydantic import BaseModel, Field


class TrainingConfig(BaseModel):
    """Fine-tuning hyperparameters and hardware configuration."""

    learning_rate: float = Field(default=1e-5, description="Initial learning rate")
    batch_size: int = Field(default=4, description="Per-device training batch size")
    num_epochs: int = Field(default=3, description="Number of training epochs")
    max_seq_length: int = Field(default=512, description="Maximum sequence length")
    gradient_accumulation_steps: int = Field(
        default=4, description="Steps for gradient accumulation"
    )
    quantization: str = Field(
        default="4bit", description="Quantization precision (4bit, 8bit, none)"
    )
    lora_r: int = Field(default=8, description="LoRA rank")
    lora_alpha: int = Field(default=16, description="LoRA alpha parameter")
    lora_dropout: float = Field(default=0.05, description="LoRA dropout rate")
