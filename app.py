import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
import os
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from huggingface_hub import hf_hub_download

def load_my_model():
    model_path = hf_hub_download(
        repo_id="rajivacharya/gi-disease-model",
        filename="gi_disease_resnet_model.h5"
    )
    return load_model(model_path, compile=False)

model = load_my_model()

class_names = ['Diverticulosis', 'Neoplasm', 'Peritonitis', 'Ureters']

def get_gradcam_heatmap(model, img_array, last_conv_layer_name="conv5_block3_out"):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    return heatmap.numpy()

st.title("🧠 GI Disease Detection (AI)")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image)

    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array)
    pred_class = np.argmax(predictions)
    confidence = np.max(predictions)

    st.write(f"Prediction: {class_names[pred_class]}")
    st.write(f"Confidence: {confidence:.2f}")

    heatmap = get_gradcam_heatmap(model, img_array)

    img_cv = np.array(image.resize((224, 224)))
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = heatmap * 0.4 + img_cv
    st.image(superimposed_img.astype("uint8"))
