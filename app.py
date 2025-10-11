import gradio as gr
import tensorflow as tf
import numpy as np
import joblib
from PIL import Image
import os
import io

# --- 1. CONFIGURATION ---
IMAGE_SIZE = (224, 224) 
CHANNELS = 3 
# Threshold for displaying only the top class (e.g., 99.999% or higher)
CONFIDENCE_THRESHOLD = 0.99999 
# NEW: Minimum confidence required for a prediction to be considered valid (20%)
MIN_ACCEPTANCE_THRESHOLD = 0.20 

# --- 2. ASSET LOADING ---
print("Attempting to load Keras model...")
try:
    model = tf.keras.models.load_model('final_plant_classifier.h5')
    print("Model loaded successfully.")
except Exception as e:
    print(f"FATAL ERROR loading model: {e}")
    model = None 

print("Attempting to load class labels...")
try:
    class_labels = joblib.load('class_labels.pkl')
    LABEL_NAMES = [class_labels[i] for i in sorted(class_labels.keys())]
    print(f"Labels loaded successfully. Found {len(LABEL_NAMES)} classes.")
except Exception as e:
    print(f"FATAL ERROR loading labels: {e}")
    LABEL_NAMES = ["Error: Labels Not Found", "Check class_labels.pkl"] 

# --- 3. INFERENCE FUNCTION with IF/ELIF/ELSE LOGIC ---
def classify_plant(input_image):
    """
    Takes a PIL Image and runs the deep learning classification, returning 
    the appropriate result based on confidence thresholds.
    """
    if model is None:
        # Return a dictionary with an error message so gr.Label displays it.
        return {"ERROR: Model failed to load on startup.": 1.0}

    # --- Preprocessing Steps ---
    img_array = np.asarray(input_image, dtype=np.float32)
    img = Image.fromarray(np.uint8(img_array)).resize(IMAGE_SIZE)
    img_array = np.asarray(img, dtype=np.float32)
    normalized_image = img_array / 255.0 
    input_tensor = np.expand_dims(normalized_image, axis=0)

    # --- Prediction ---
    predictions = model.predict(input_tensor)
    probabilities = predictions[0] 
    
    # Get the index of the highest prediction
    top_index = np.argmax(probabilities)
    top_confidence = probabilities[top_index]
    top_label = LABEL_NAMES[top_index]
    
    # Format all results as a list of tuples: [(label, confidence), ...]
    all_results_list = sorted(zip(LABEL_NAMES, probabilities), key=lambda x: x[1], reverse=True)
    
    # Convert the full list of results into a dictionary required by gr.Label
    full_results_dict = dict(all_results_list)

    # --- CONDITIONAL RETURN LOGIC (IF/ELIF/ELSE) ---
    
    # 1. NOT FOUND CHECK
    # If the top prediction is below the minimum required confidence (MIN_ACCEPTANCE_THRESHOLD = 0.20)
    if top_confidence < MIN_ACCEPTANCE_THRESHOLD:
        # Gradio Label displays the key as the label and the value as the score.
        # We use a 1.0 score here to make the "Not Found" bar visually full.
        return {"NOT FOUND: Could not identify a match (confidence too low).": 1.0}
    
    # 2. 100% MATCH CHECK (Existing Logic)
    # If the top confidence is very high, return only that one label-score pair.
    elif top_confidence >= CONFIDENCE_THRESHOLD:
        return {top_label: top_confidence} 
    
    # 3. DEFAULT (Show Top 3)
    # Otherwise, return the full dictionary, allowing gr.Label to display the top 3.
    else:
        return full_results_dict


# --- 4. GRADIO INTERFACE SETUP (No change needed here) ---

image_input = gr.Image(type="pil", label="Upload Leaf Image")

# num_top_classes=3 ensures we only see the top 3 when the full dictionary is returned.
# If we return a dictionary with only one item (Case 1 or 2), only that item is shown.
label_output = gr.Label(num_top_classes=3, label="Predicted Medicinal Plant")

title = "Title: Scan Your Medicinal Leaf with AI"
description = (
    "Upload an image of a medicinal plant leaf. If the model cannot confidently identify the plant, "
    "it will show 'NOT FOUND.' Otherwise, it displays either the single high-confidence result or the top 3 predictions. "
    "Training is set for 40 epochs; model will be highly accurate!"
)

iface = gr.Interface(
    fn=classify_plant,
    inputs=image_input,
    outputs=label_output, 
    title=title,
    description=description,
    theme=gr.themes.Soft(),
    allow_flagging="auto", 
    live=True 
)

iface.launch()
