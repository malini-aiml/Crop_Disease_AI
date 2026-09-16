import streamlit as st
import ai_edge_litert.interpreter as tflite
from PIL import Image, ImageOps
import numpy as np

st.set_page_config(page_title="Crop Disease Detector", page_icon="🌿")

st.title("🌿 Crop Disease Detection AI App")
st.write("Upload a leaf image to identify potential plant diseases and get treatment solutions.")

@st.cache_resource
def load_tflite_model():
    interpreter = tflite.Interpreter(model_path="crop_disease_model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

uploaded_file = st.file_uploader("Upload Leaf Image (JPG/PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Crop Image", use_column_width=True)
    
    st.write("🔍 **AI Model Analyzing Image...**")
    
    # Preprocess image
    size = (128, 128)
    image_resized = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image_resized, dtype=np.float32) / 255.0
    img_reshape = np.expand_dims(img_array, axis=0)
    
    # TFLite Prediction
    interpreter.set_tensor(input_details[0]['index'], img_reshape)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])
    
    classes = ["Healthy Leaf 🟢", "Early Blight Disease 🟡", "Late Blight Disease 🔴"]
    result = classes[np.argmax(prediction)]
    
    st.success(f"**Diagnosis Result:** {result}")
    st.markdown("---")
    st.subheader("💡 Recommended Solution:")
    st.info("Ensure adequate sunlight, avoid over-watering, and spray neem oil if leaf spots appear.")
