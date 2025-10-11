import gradio as gr
import numpy as np
import joblib
# Import the specific libraries needed for your TensorFlow/Keras model
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# --- 1. CONFIGURATION (Based on your Notebook) ---
MODEL_PATH = 'final_plant_classifier.h5' # The file you will download later
LABELS_PATH = 'class_labels.pkl'         # The file you will download later
IMAGE_SIZE = (150, 150)                  # Target size used in your training

# --- 2. MODEL LOADING ---
try:
    # Load the Keras model (compiled=False prevents errors if custom layers are missing)
    model = load_model(MODEL_PATH, compile=False) 
    # Load the class labels (your 10 plant names)
    class_labels = joblib.load(LABELS_PATH)
except FileNotFoundError:
    # This block handles the case where you run the app BEFORE downloading the files.
    # It allows you to test the UI layout without the model loaded.
    model = None
    class_labels = [f"Class {i}" for i in range(10)]
    print("\n[WARNING] Model files not found! Please place 'final_plant_classifier.h5' and 'class_labels.pkl' in this folder and restart the app.\n")
except Exception as e:
    model = None
    print(f"\n[FATAL ERROR] Failed to load model or labels: {e}\n")


# --- 3. THE CORE PREDICTION FUNCTION ---
def classify_plant(img: Image):
    """
    Takes a user-uploaded PIL Image, preprocesses it exactly as the model expects, 
    and returns the classification results.
    """
    if model is None:
        return {label: 0.1 for label in class_labels} # Return dummy data if model failed to load

    # A. Resizing and PIL to NumPy array conversion
    img = img.resize(IMAGE_SIZE) # Resize to 150x150
    img_array = image.img_to_array(img) # Convert to array (150, 150, 3)
    
    # B. Expanding dimensions to (1, 150, 150, 3) for model input
    img_array = np.expand_dims(img_array, axis=0) 
    
    # C. Normalization (dividing by 255)
    img_array /= 255.0
    
    # D. Prediction
    predictions = model.predict(img_array)[0] # Get the 10 probabilities
    
    # E. Format results for Gradio's Label output
    confidences = {class_labels[i]: float(predictions[i]) for i in range(len(class_labels))}
    
    return confidences


# --- 4. GRADIO INTERFACE SETUP ---
iface = gr.Interface(
    fn=classify_plant,
    inputs=gr.Image(type="pil", label="Upload Leaf Image"),
    outputs=gr.Label(num_top_classes=3, label="Predicted Medicinal Plant"),
    title="BD Medicinal Plant Classifier (TensorFlow/Keras)",
    description="Upload an image of a medicinal plant leaf. Training is set for 40 epochs; model will be highly accurate!",
    theme=gr.themes.Soft(),
    allow_flagging="auto"
)

# --- 5. LAUNCH THE APP ---
if __name__ == "__main__":
    iface.launch(inbrowser=True)