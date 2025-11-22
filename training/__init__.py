# -*- coding: utf-8 -*-
"""TARS training configuration"""

import os
import sys

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.tars_agent import TARS
from configs.training_config import TrainingConfig
