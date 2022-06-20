import numpy
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image
import tensorflow as tf

from keras_words import process_image, preprocess_image, decode_batch_predictions
from keras_words import WordsModel


def remove_empty_columns(img: numpy.ndarray):
    pad_value = 0
    rimg = img.sum(-1)
    arr_masked = np.all(rimg == pad_value, axis=1)
    pad_top = np.argmin(arr_masked, axis=0)
    pad_south = arr_masked.shape[0] - np.argmin(arr_masked[::-1], axis=0)
    img = img[pad_top:pad_south, :, :]
    arr_masked = np.all(rimg == pad_value, axis=0)
    pad_left = np.argmin(arr_masked, axis=0)
    pad_right = arr_masked.shape[0] - np.argmin(arr_masked[::-1], axis=0)
    img = img[:, pad_left:pad_right, :]
    return img


st.title("Draw")

canvas_result = st_canvas(width=128*5, height=32*10, stroke_color='blue', stroke_width=8)

if canvas_result.image_data is not None:
    img: np.ndarray = canvas_result.image_data
    img = remove_empty_columns(img)
    print(img)
    fixed = preprocess_image(img)
    fixed = tf.image.flip_left_right(fixed)
    fixed = tf.transpose(fixed, perm=[1, 0, 2])
    fixed = fixed.numpy()[:,:,0]
    print(fixed.shape)
    st.image(fixed, output_format='png')
    x = process_image([img])
    pred = WordsModel.predict(x)
    text = decode_batch_predictions(pred)
    st.title(f"Predicted text: {text[0]}")
