"""
Implements original transformer model.
"""

import tensorflow as tf
from dataclasses import dataclass
from typing import Union


@dataclass
class ModelHp:
    d_model: int
    num_attention_heads: int
    dropout_rate: Union[float, None]
    num_units: int
    activation: str
    vocab_size: int
    num_layers: int

