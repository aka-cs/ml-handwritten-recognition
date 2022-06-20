import json
from pathlib import Path

import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers.experimental.preprocessing import StringLookup


with open(f"{Path(__file__).parent}/models/vocab.json", "r") as f:
    vocabulary = json.load(f)

max_len = 21

# Mapping characters to integers.
char_to_num = StringLookup(vocabulary=list(vocabulary), mask_token=None)

# Mapping integers back to original characters.
num_to_char = StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)


def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search.
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
        :, :max_len
    ]
    # Iterate over the results and get back the text.
    output_text = []
    for res in results:
        res = tf.gather(res, tf.where(tf.math.not_equal(res, -1)))
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text