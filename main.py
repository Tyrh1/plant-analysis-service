from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import cv2
import pickle
import base64
import io
from PIL import Image
import os
import requests
import zipfile

app = FastAPI()

class AnalysisRequest(BaseModel):
    image_base64: str
    plant_threshold: float = 0.5
    classification_threshold: float = 0.6

class PlantAnalyzer:
    def __init__(self):
        self.classifier_model = None
        self.detector_model = None
        self.label_encoder = None
        self.load_models()
    
    def download_models(self):
        """Download models from cloud storage if not present locally"""
        models_dir = 'models'
        os.makedirs(models_dir, exist_ok=True)
        
        model_files = {
            'best_plant_classifier_advanced.h5': os.environ.get('CLASSIFIER_MODEL_URL'),
            'esp32_plant_detector.h5': os.environ.get('DETECTOR_MODEL_URL'), 
            'label_encoder_advanced.pkl': os.environ.get('LABEL_ENCODER_URL')
        }
        
        fallback_urls = {
            'best_plant_classifier_advanced.h5': 'https://huggingface.co/tyrh1/plant-analysis-models/resolve/main/best_plant_classifier_advanced.h5',
            'esp32_plant_detector.h5': 'https://huggingface.co/tyrh1/plant-analysis-models/resolve/main/esp32_plant_detector.h5',
            'label_encoder_advanced.pkl': 'https://huggingface.co/tyrh1/plant-analysis-models/resolve/main/label_encoder_advanced.pkl'
        }
        
        for filename, url in model_files.items():
            if not url:
                url = fallback_urls[filename]
                
            filepath = os.path.join(models_dir, filename)
            if not os.path.exists(filepath):
                try:
                    print(f"Downloading {filename}...")
                    
                    response = requests.get(url, stream=True, timeout=300)
                    
                    content_type = response.headers.get('content-type', '')
                    if 'text/html' in content_type and 'huggingface.co' in url:
                        print(f"❌ {filename}: File not found on Hugging Face Hub")
                        print("💡 Solution: Upload your model files to https://huggingface.co/tyrh1/plant-analysis-models")
                        print(f"   Expected URL: {url}")
                        continue
                    elif 'text/html' in content_type:
                        print(f"❌ {filename}: Server serving HTML instead of file")
                        print("💡 Solution: Upload files to Hugging Face Hub or GitHub Releases")
                        continue
                    
                    response.raise_for_status()
                    
                    # Download file
                    with open(filepath, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                    
                    # Verify file size
                    file_size = os.path.getsize(filepath)
                    if file_size < 1000:
                        print(f"❌ {filename}: File too small ({file_size} bytes), likely corrupted")
                        os.remove(filepath)
                        continue
                        
                    print(f"✅ {filename}: Downloaded successfully ({file_size:,} bytes)")
                    
                except Exception as e:
                    print(f"❌ Error downloading {filename}: {e}")
                    print("💡 Upload your models to: https://huggingface.co/tyrh1/plant-analysis-models")

    def load_models(self):
        """Load the trained CNN models"""
        try:
            self.download_models()
            
            # Load models (you'll need to upload these to your service)
            self.classifier_model = tf.keras.models.load_model('models/best_plant_classifier_advanced.h5')
            self.detector_model = tf.keras.models.load_model('models/esp32_plant_detector.h5')
            
            with open('models/label_encoder_advanced.pkl', 'rb') as f:
                self.label_encoder = pickle.load(f)
                
            print("Models loaded successfully")
        except Exception as e:
            print(f"Error loading models: {e}")
    
    def preprocess_image_from_base64(self, base64_string, target_size=(224, 224)):
        """Convert base64 image to preprocessed array"""
        try:
            # Decode base64
            image_data = base64.b64decode(base64_string)
            image = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert to numpy array
            img_array = np.array(image)
            
            # Resize
            img_array = cv2.resize(img_array, target_size)
            
            # Normalize
            img_array = img_array.astype(np.float32) / 255.0
            
            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)
            
            return img_array
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error preprocessing image: {e}")
    
    def analyze_image(self, image_array, plant_threshold=0.5, classification_threshold=0.6):
        """Analyze plant image using CNN models"""
        
        # Step 1: Plant detection
        detection_pred = self.detector_model.predict(image_array, verbose=0)
        plant_confidence = detection_pred[0][0]
        
        if plant_confidence <= plant_threshold:
            return {
                'status': 'invalid',
                'message': 'No plant detected in the image',
                'plant_detected': False,
                'detection_confidence': float(plant_confidence)
            }
        
        # Step 2: Plant classification
        classification_pred = self.classifier_model.predict(image_array, verbose=0)
        predicted_class_idx = np.argmax(classification_pred[0])
        confidence = np.max(classification_pred[0])
        
        if confidence < classification_threshold:
            return {
                'status': 'invalid',
                'message': f'Low classification confidence: {confidence:.4f}',
                'plant_detected': True,
                'detection_confidence': float(plant_confidence),
                'classification_confidence': float(confidence)
            }
        
        # Decode prediction
        predicted_class = self.label_encoder.inverse_transform([predicted_class_idx])[0]
        
        # Parse results
        if 'Eggplant' in predicted_class:
            plant_type = 'Eggplant'
        elif 'Tomato' in predicted_class:
            plant_type = 'Tomato'
        else:
            plant_type = 'Unknown'
        
        if 'Healthy' in predicted_class:
            health_status = 'Healthy'
        elif 'Unhealthy' in predicted_class:
            health_status = 'Unhealthy'
        else:
            health_status = 'Unknown'
        
        return {
            'status': 'valid',
            'plant_detected': True,
            'plant_type': plant_type,
            'health_status': health_status,
            'confidence': float(confidence),
            'detection_confidence': float(plant_confidence),
            'full_class': predicted_class
        }

# Initialize analyzer
analyzer = PlantAnalyzer()

@app.get("/")
async def root():
    """Root endpoint to handle health checks and provide service info"""
    return {
        "service": "Plant Analysis Service",
        "status": "running",
        "models_loaded": analyzer.classifier_model is not None and analyzer.detector_model is not None,
        "endpoints": ["/analyze", "/health"]
    }

@app.post("/analyze")
async def analyze_plant(request: AnalysisRequest):
    """Analyze plant image endpoint"""
    
    if not analyzer.classifier_model or not analyzer.detector_model:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    try:
        # Preprocess image
        image_array = analyzer.preprocess_image_from_base64(request.image_base64)
        
        # Analyze
        result = analyzer.analyze_image(
            image_array, 
            request.plant_threshold, 
            request.classification_threshold
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "models_loaded": analyzer.classifier_model is not None}

@app.on_event("startup")
async def startup_event():
    """Log startup information and model status"""
    print("=== Plant Analysis Service Starting ===")
    if analyzer.classifier_model and analyzer.detector_model and analyzer.label_encoder:
        print("✅ All models loaded successfully!")
    else:
        print("❌ Model loading failed - check download logs above")
        print(f"Classifier: {'✅' if analyzer.classifier_model else '❌'}")
        print(f"Detector: {'✅' if analyzer.detector_model else '❌'}")
        print(f"Label Encoder: {'✅' if analyzer.label_encoder else '❌'}")
        print()
        print("🔧 SOLUTIONS:")
        print("1. Upload models to Hugging Face Hub: https://huggingface.co/")
        print("2. Create repository: tyrh1/plant-analysis-models")
        print("3. Upload: best_plant_classifier_advanced.h5, esp32_plant_detector.h5, label_encoder_advanced.pkl")
        print("4. Files should be accessible at the URLs shown in download logs above")
    print("=== Service Ready ===")
