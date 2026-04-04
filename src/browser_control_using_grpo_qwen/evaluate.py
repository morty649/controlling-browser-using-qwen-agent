from envs.browsergym_env import BrowserGymEnv,BrowserGymAction
from transformers import AutoModelForCausalLM,AutoTokenizer

import numpy as np 
from PIL import image
import os

from .configuration import FineTuneConfig
from .paths import get_path_to_media

config = FineTuneConfig.from_yaml(file_name='qwen_evalution_debug.yaml') 
system_prompt = config.system_prompt
max_steps = config.max_steps


# Let us write this after some time

