# Flow-GRPO Package
# This file makes flow_grpo a Python package

__version__ = "0.1.0"
__author__ = "Flow-RTPO Team"

# Import key modules for easier access
from .stat_tracking import PerPromptStatTracker
from .prompt_editor import PromptEditorPolicy
from .toxicity_rewards import toxicity_reward_function
from .rtp_dataset import create_rtp_dataset_and_loader
from .ema import EMAModuleWrapper

__all__ = [
    "PerPromptStatTracker",
    "PromptEditorPolicy", 
    "toxicity_reward_function",
    "create_rtp_dataset_and_loader",
    "EMAModuleWrapper"
] 