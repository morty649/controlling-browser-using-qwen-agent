from typing import Any, Optional

from datetime import datetime
from pathlib import Path
from typing import Self

import yaml
from pydantic import model_validator
from pydantic_settings import BaseSettings
  
from .paths import get_path_to_configs

class FineTuneConfig(BaseSettings):
    seed : int
    resume_from_checkpoint : Optional[str] = None # Resumes training from checkpoint if provided

    # Language model parameters
    model_name : str # HF model name
    max_seq_length : str # Number of tokens in context window for the LM
    system_prompt : str # System prompt it is

    #BrowserGym environment 
    browsergym_url : str # where it is available
    dataset_size : str # from browsergym the no. of scenarios we use  
    default_goal : str # if the environment does not provide more explicit instructions

    # How long and how aggressively we train
    learning_rate : float
    warmup_steps : int

    #vLLM inference
    max_steps : int # once reset how many steps before stopping
    per_device_train_batch_size : int # no. of samples per device per step
    num_generations : int #no. of completions to generate per prompt
    generations_batch_size : int # batch size used during generation
    max_completion_length : int
    use_vllm : bool # use vLLM engine for faster inference
    vllm_mode : str # vLLM mode : colocate , runs generation on the same GPU as training
    vllm_gpu_memory_utilization : float # how much fraction of GPU should the vLLM should use

    # experiment tracking using wandb
    wandb_enabled : bool 
    wandb_project_name : str 
    wandb_experiment_name : str | None = None
    logging_steps : int  # How often do we print our training loss
    push_to_hf : Optional[bool] = True

    # LoRA specific hyperparameters for parameter efficient finetuning
    use_peft: bool = False # Default : disabled for backward compatibility
    lora_r : int = 8 # LoRA rank : memory efficient default
    lora_alpha : int = 16 # 2* rank , lora scaling factor
    lora_dropout : float  = 0.0
    lora_bias : str = "none"
    use_rslora : bool = False
    lora_target_modules : list[str] = [
        "q_proj","k_proj","o_proj","v_proj",
        "gate_proj","up_proj","down_proj",
    ]

    # max_steps: int = 10000  # can be increased if i have modal credits but for now
    # save_steps: int = 1000  # increase
    # eval_steps: int = 1000  # increase
    # eval_sample_callback_enabled: bool = False

    @classmethod
    def from_yaml(cls,file_name:str)->Self:
        '''Loads configuration from a yaml file located in configuration_files'''

        file_path = str(Path(get_path_to_configs()) / file_name)
        print(f"Loading configs from {file_path}")
        with open(file_path) as f:
            data = yaml.safe_load(f)

        print(f"Loaded data : {data}")

        return cls(**data)
    
    @model_validator(mode="after")
    def set_experiment_name(self):
        if self.wandb_experiment_name is None: 
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            model_short = self.model_name.split("/")[-1]
            self.wandb_experiment_name = (
                f"{model_short}-browsergym-{timestamp}"
            )

        return self
    



