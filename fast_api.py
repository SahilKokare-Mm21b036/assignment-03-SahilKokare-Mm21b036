from fastapi import FastAPI, UploadFile, File, HTTPException
import numpy as np
from PIL import Image
from io import BytesIO
import pickle

app = FastAPI()

def load_model(filename):
    """Loads a pickled model from the given filename."""
    try:
        with open(filename, 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Model file '{filename}' not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {e}")

model = load_model("final_model.pkl")  # Load the model on startup

@app.post("/predict")
async def predict_digit(file: UploadFile = File(...)):
    
    try:
        image_data = await file.read()
        image = Image.open(BytesIO(image_data)).convert('L')
        image = image.resize((28, 28))
        image_array = np.array(image)
        flattened_vector = image_array.flatten() / 255.0

        
        prediction = model.predict(flattened_vector.reshape(1, -1))

        
        return {"predicted_digit": int(prediction)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {e}")
    
    
