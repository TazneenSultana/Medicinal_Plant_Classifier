import gradio as gr
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image, ImageFilter # ImageFilter restored for heuristic
import joblib
import os
from tensorflow.keras.layers import InputLayer 
from tensorflow.keras.utils import custom_object_scope # Required for the most robust fix

# =========================================================
# 1. CONFIGURATION CONSTANTS
# =========================================================
MIN_ACCEPTANCE_THRESHOLD = 0.10  
NEGLIGIBLE_CONFIDENCE = 0.001  
DEBUG = False  

# =========================================================
# 2. MODEL AND LABEL LOADING
# =========================================================
MODEL_PATH = "final_plant_classifier.h5"
LABELS_PATH = "class_labels.pkl"
MODEL = None
LABEL_NAMES = None

# --- CRITICAL FIX START: Resolving 'batch_shape' & 'DTypePolicy' errors ---

class Identity(object):
    """
    A dummy class to absorb serialization errors for legacy custom objects 
    (like DTypePolicy) that newer TF versions don't recognize.
    """
    def __init__(self, *args, **kwargs):
        pass
    
    # CRITICAL FIX: Absorb all possible arguments passed by the loader 
    # to prevent "missing positional argument" errors during deserialization.
    def from_config(self, *args, **kwargs):
        return self

def fix_input_layer_config(**kwargs):
    """
    Intercept function for InputLayer. It removes the legacy 'batch_shape' 
    argument and replaces it with the expected 'input_shape' format.
    """
    config = kwargs.get('config', {})
    
    # Handle legacy batch_shape argument passed outside of config
    if 'batch_shape' in kwargs and 'batch_shape' not in config:
        config['batch_shape'] = kwargs['batch_shape']
    
    if 'name' in kwargs and 'name' not in config:
        config['name'] = kwargs['name']
    
    if 'dtype' in kwargs and 'dtype' not in config:
        config['dtype'] = kwargs['dtype']

    # Apply the core fix: replace 'batch_shape' with 'input_shape'
    if 'batch_shape' in config:
        # batch_shape format: [None, H, W, C] -> input_shape format: [H, W, C]
        config['input_shape'] = config['batch_shape'][1:]
        del config['batch_shape']
        
    return InputLayer.from_config(config)

print("Attempting to load Keras model...")
try:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    custom_objects = {
        'InputLayer': fix_input_layer_config,
        'DTypePolicy': Identity # Fixes the Conv2D deserialization error
    }
    
    # Use custom_object_scope for the most reliable way to apply these fixes globally
    with custom_object_scope(custom_objects):
        MODEL = load_model(
            MODEL_PATH, 
            compile=False
        )
    print("Model loaded successfully.")
except Exception as e:
    print(f"FATAL ERROR loading model: {e}")
# --- CRITICAL FIX END ---

print("Attempting to load class labels...")
try:
    class_labels = joblib.load(LABELS_PATH)
    if isinstance(class_labels, (list, np.ndarray)):
        LABEL_NAMES = list(class_labels)
    elif isinstance(class_labels, dict):
        try:
            LABEL_NAMES = [class_labels[i] for i in sorted(class_labels.keys())]
        except Exception:
            LABEL_NAMES = list(class_labels.values())
    else:
        raise TypeError("Loaded label file is neither a list/array nor a dictionary.")
    print(f"Labels loaded successfully. Found {len(LABEL_NAMES)} classes.")
except Exception as e:
    print(f"FATAL ERROR loading labels: {e}")

# =========================================================
# 3. BASIC NON-LEAF DETECTION 
# =========================================================
def looks_like_leaf(image: Image.Image) -> bool:
    """
    Simple heuristic to check if the uploaded image looks like a leaf.
    Returns False if it's probably not a leaf.
    """
    if image is None:
        return True # Assume OK if no image provided yet
        
    img = image.resize((128, 128)).convert("RGB")
    img_np = np.array(img) / 255.0

    # Calculate color statistics
    mean_rgb = img_np.mean(axis=(0, 1))
    green_dominant = mean_rgb[1] > mean_rgb[0] + 0.05 and mean_rgb[1] > mean_rgb[2] + 0.05

    # Texture detection using edge variance
    edges = img.filter(ImageFilter.FIND_EDGES)
    edge_array = np.array(edges.convert("L")) / 255.0
    texture_score = edge_array.std()

    # Brightness check
    brightness = img_np.mean()

    if DEBUG:
        print(f"Mean RGB: {mean_rgb}, Green Dominant: {green_dominant}")
        print(f"Texture Std: {texture_score:.4f}, Brightness: {brightness:.4f}")

    # Basic rule-based decision
    if not green_dominant and brightness > 0.6:
        return False  # too bright, not leaf-like
    if texture_score < 0.05:
        return False  # too smooth or blank image
    if brightness < 0.05:
        return False  # too dark

    return True

# =========================================================
# 4. IMAGE PREPROCESSING
# =========================================================
def preprocess_image(image: Image.Image) -> np.ndarray:
    if image is None:
        return np.array([])
    image = image.convert("RGB")
    
    # Robustly check if MODEL is loaded before accessing input_shape
    if MODEL and len(MODEL.input_shape) >= 2:
        img_size = MODEL.input_shape[1]
    else:
        # Fallback if model loading failed or input shape is undefined
        img_size = 224 
        
    image = image.resize((img_size, img_size))
    img_array = img_to_array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# =========================================================
# 5. CLASSIFICATION FUNCTION
# =========================================================
def classify_plant(image: Image.Image):
    default_update = gr.update(value={}, label="Prediction Results", num_top_classes=3)

    # Step 1: Sanity check (User's OOD Heuristic)
    if 'ImageFilter' in globals() and not looks_like_leaf(image):
        return gr.update(value={"Not a leaf / invalid image": 1.0},
                            label="Prediction Status (Heuristic)", num_top_classes=1)

    # Step 2: Model and label check
    if MODEL is None or LABEL_NAMES is None:
        return gr.update(value={"Model Error": 1.0},
                            label="Prediction Status", num_top_classes=1)

    processed_image = preprocess_image(image)
    if processed_image.size == 0:
        return default_update

    predictions = MODEL.predict(processed_image, verbose=0)[0]
    predictions = np.clip(predictions, 0.0, 1.0)
    predictions = predictions / predictions.sum()
    top_indices = np.argsort(predictions)[::-1]

    top_confidence = predictions[top_indices[0]]
    second_confidence = predictions[top_indices[1]] if len(top_indices) > 1 else 0.0
    third_confidence = predictions[top_indices[2]] if len(top_indices) > 2 else 0.0

    if DEBUG:
        print("\n--- DEBUGGING CONFIDENCE SCORES ---")
        print(f"Top Confidence: {top_confidence:.15f}")
        print(f"Second Confidence: {second_confidence:.15f}")
        print(f"Third Confidence: {third_confidence:.15f}")
        print("------------------------------------\n")

    # RULE 1: HIGH CONFIDENCE SINGLE CLASS
    if (
        top_confidence >= 0.999 or
        (second_confidence < NEGLIGIBLE_CONFIDENCE and third_confidence < NEGLIGIBLE_CONFIDENCE)
    ):
        if top_confidence < MIN_ACCEPTANCE_THRESHOLD:
            return gr.update(value={"NOT FOUND / Low Confidence": 1.0},
                             label="Prediction Status", num_top_classes=1)
        top_class_name = LABEL_NAMES[top_indices[0]]
        return gr.update(value={top_class_name: float(top_confidence)},
                         label="Predicted Medicinal Plant (Single Class)", num_top_classes=1)

    # RULE 2: LOW CONFIDENCE REJECTION
    elif top_confidence < MIN_ACCEPTANCE_THRESHOLD:
        return gr.update(value={"NOT FOUND / Low Confidence": 1.0},
                         label="Prediction Status", num_top_classes=1)

    # RULE 3: NORMAL TOP-3 DISPLAY
    else:
        results = {}
        num_to_show = min(3, len(LABEL_NAMES))
        for i in range(num_to_show):
            index = top_indices[i]
            name = LABEL_NAMES[index]
            score = predictions[index]
            results[name] = float(score)
        return gr.update(value=results, label="Predicted Medicinal Plant (Top 3)", num_top_classes=3)

# =========================================================
# 6. GRADIO INTERFACE
# =========================================================
with gr.Blocks(theme=gr.themes.Soft(primary_hue="green"), title="Medicinal Plant Classifier") as iface:
    gr.Markdown("# 🌱 Scan Your Medicinal Leaf with AI")
    #gr.Markdown("Upload an image of a medicinal plant leaf. This model was trained for 40 epochs.")

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Upload Leaf Image")
            with gr.Row():
                submit_button = gr.Button("Submit", variant="primary")
                clear_button = gr.Button("Clear", variant="secondary")

        with gr.Column(scale=1):
            final_output_label = gr.Label(
                num_top_classes=3, label="Prediction Results", value={}, visible=True
            )

    submit_button.click(fn=classify_plant, inputs=image_input, outputs=final_output_label)

    def clear_all():
        return [None, gr.update(value={}, label="Prediction Results", num_top_classes=3)]

    clear_button.click(fn=clear_all, inputs=[], outputs=[image_input, final_output_label])

# =========================================================
# 7. LAUNCH APP
# =========================================================
if __name__ == "__main__":
    iface.launch(share=False)
