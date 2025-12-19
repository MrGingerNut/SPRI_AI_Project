#!/usr/bin/env python3
"""
Ejemplo para usar el modelo WildFireNet entrenado
"""
import torch
import torch.nn.functional as F
from wildfire_model import WildFireNet2DV3L_3x3_Residual as WildFireNet
import numpy as np

def load_trained_model(model_path="Wildfire_N__128__AUG_1_valid_best_3264.pth"):
    """Carga el modelo entrenado"""
    # Configuración IDÉNTICA al entrenamiento
    dims = (32, 64)  # ⚠️ IMPORTANTE: usar las mismas dimensiones
    
    # Crear modelo
    model = WildFireNet(input_channels=4, num_classes=2, dims=dims)
    
    # Cargar pesos
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint)
    model.eval()
    
    print(f"✅ Modelo cargado: {model_path}")
    return model

def preprocess_image(image_array):
    """
    Preprocesa una imagen para el modelo.
    image_array: numpy array de forma (H, W, 4) o (4, H, W)
    """
    # Convertir a tensor
    if isinstance(image_array, np.ndarray):
        tensor = torch.from_numpy(image_array).float()
    else:
        tensor = image_array
    
    # Asegurar formato (batch, channels, height, width)
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)  # Añadir dimensión batch
    
    # Normalizar si es necesario (ajustar según tu dataset)
    # tensor = (tensor - mean) / std
    
    return tensor

def predict(model, input_tensor):
    """Realiza predicción"""
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)
        
        # Obtener clase predicha
        _, predicted = torch.max(probabilities, 1)
    
    return predicted, probabilities

def process_single_image(model, image_path):
    """
    Procesa una sola imagen.
    Nota: Necesitas cargar tu imagen multiespectral (4 canales)
    """
    # Ejemplo con imagen dummy
    print("\n🧪 Ejemplo con imagen dummy:")
    
    # Crear imagen dummy (4 canales, 128x128)
    dummy_image = torch.randn(1, 4, 128, 128)
    
    # Hacer predicción
    prediction, probs = predict(model, dummy_image)
    
    print(f"   Forma de entrada: {dummy_image.shape}")
    print(f"   Predicción shape: {prediction.shape}")
    print(f"   Probabilidades shape: {probs.shape}")
    print(f"   Prob clase 0: {probs[0, 0].mean():.4f}")
    print(f"   Prob clase 1: {probs[0, 1].mean():.4f}")
    
    return prediction, probs

def batch_inference(model, batch_size=4):
    """Ejemplo de inferencia por lotes"""
    print(f"\n📦 Inferencia por lotes (batch={batch_size}):")
    
    # Crear batch dummy
    batch = torch.randn(batch_size, 4, 128, 128)
    
    # Predecir
    predictions, probabilities = predict(model, batch)
    
    # Estadísticas
    fire_pixels = (predictions == 1).float().mean().item()
    
    print(f"   Batch procesado: {batch_size} imágenes")
    print(f"   Porcentaje de píxeles 'incendio': {fire_pixels*100:.2f}%")
    print(f"   Probabilidad media incendio: {probabilities[:, 1].mean():.4f}")
    
    return predictions, probabilities

if __name__ == "__main__":
    # 1. Cargar modelo (elige uno)
    MODEL_PATH = "Wildfire_N__128__AUG_1_valid_best_3264.pth"  # Mejor F1-score
    # MODEL_PATH = "Wildfire_N__128__AUG_1_best.pth"  # Mejor loss
    
    model = load_trained_model(MODEL_PATH)
    
    # 2. Probar con una imagen
    pred, probs = process_single_image(model, "dummy")
    
    # 3. Probar batch
    batch_pred, batch_probs = batch_inference(model, batch_size=4)
    
    print("\n✅ Ejemplo completado exitosamente")
    print("\n📝 Para usar con tus propias imágenes:")
    print("   1. Carga tus imágenes multiespectrales (4 canales)")
    print("   2. Redimensiona a 128x128 si es necesario")
    print("   3. Normaliza igual que durante el entrenamiento")
    print("   4. Usa la función predict()")
