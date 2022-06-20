import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from .ctc import CTCLayer

np.random.seed(42)
tf.random.set_seed(42)


_model = keras.models.load_model(f"{Path(__file__).parent}/models/wr_model.h5", custom_objects={'CTCLayer': CTCLayer})

WordsModel = keras.models.Model(
    _model.get_layer(name="image").input, _model.get_layer(name="dense2").output
)

