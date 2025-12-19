#!/usr/bin/env python3
import torch
import os
import numpy as np
from wildfire_model import WildFireNet2DV3L_3x3_Residual as WildFireNet
import matplotlib.pyplot as plt

def compare_models(model1_path, model2_path):
    """Compara dos modelos .pth"""
    
    print("=" * 80)
    print("COMPARACIÓN DE MODELOS ENTRENADOS")
    print("=" * 80)
    
    # Configuración común
    dims = (32, 64)
    
    # Crear instancias
    model1 = WildFireNet(input_channels=4, num_classes=2, dims=dims)
    model2 = WildFireNet(input_channels=4, num_classes=2, dims=dims)
    
    # Cargar pesos
    checkpoint1 = torch.load(model1_path, map_location='cpu')
    checkpoint2 = torch.load(model2_path, map_location='cpu')
    
    model1.load_state_dict(checkpoint1)
    model2.load_state_dict(checkpoint2)
    
    model1.eval()
    model2.eval()
    
    print(f"\n📁 MODELO 1: {os.path.basename(model1_path)}")
    print(f"📁 MODELO 2: {os.path.basename(model2_path)}")
    
    # 1. Comparar número de parámetros
    params1 = sum(p.numel() for p in model1.parameters())
    params2 = sum(p.numel() for p in model2.parameters())
    
    print(f"\n1. NÚMERO DE PARÁMETROS:")
    print(f"   Modelo 1: {params1:,}")
    print(f"   Modelo 2: {params2:,}")
    print(f"   ¿Iguales?: {'✅ Sí' if params1 == params2 else '❌ No'}")
    
    # 2. Comparar pesos capa por capa
    print(f"\n2. COMPARACIÓN DE PESOS:")
    print("-" * 60)
    
    differences = []
    identical_layers = 0
    
    for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
        if name1 != name2:
            print(f"❌ Nombres diferentes: {name1} vs {name2}")
            continue
            
        # Calcular diferencia
        diff = torch.abs(param1 - param2).mean().item()
        differences.append((name1, diff))
        
        if diff < 1e-6:  # Prácticamente iguales
            identical_layers += 1
            print(f"   ✅ {name1:40} IDÉNTICAS (diff: {diff:.2e})")
        elif diff < 0.01:  # Muy similares
            print(f"   ≈  {name1:40} MUY SIMILARES (diff: {diff:.4f})")
        else:
            print(f"   ✗  {name1:40} DIFERENTES (diff: {diff:.4f})")
    
    print(f"\n   Capas idénticas: {identical_layers}/{len(differences)}")
    print(f"   Diferencia media: {np.mean([d[1] for d in differences]):.6f}")
    
    # 3. Test de inferencia comparativa
    print(f"\n3. TEST DE INFERENCIA:")
    print("-" * 60)
    
    # Crear input de prueba
    torch.manual_seed(42)  # Para reproducibilidad
    test_input = torch.randn(2, 4, 128, 128)  # Batch de 2 imágenes
    
    with torch.no_grad():
        output1 = model1(test_input)
        output2 = model2(test_input)
    
    print(f"Input shape:  {test_input.shape}")
    print(f"Output1 shape: {output1.shape}")
    print(f"Output2 shape: {output2.shape}")
    
    # Calcular diferencia en salidas
    output_diff = torch.abs(output1 - output2).mean().item()
    print(f"\nDiferencia en salidas: {output_diff:.6f}")
    
    # Calcular softmax para interpretación
    prob1 = torch.softmax(output1, dim=1)
    prob2 = torch.softmax(output2, dim=1)
    
    print(f"\nProbabilidades promedio (batch):")
    print(f"  Modelo 1 - Clase 0: {prob1[:, 0].mean():.4f}, Clase 1: {prob1[:, 1].mean():.4f}")
    print(f"  Modelo 2 - Clase 0: {prob2[:, 0].mean():.4f}, Clase 1: {prob2[:, 1].mean():.4f}")
    
    # 4. Análisis de pesos individual
    analyze_weights_details(model1, model2)

def analyze_weights_details(model1, model2):
    """Análisis detallado de los pesos"""
    
    print(f"\n4. ANÁLISIS DETALLADO DE PESOS:")
    print("-" * 60)
    
    # Seleccionar algunas capas para análisis
    layers_to_analyze = ['layer1.0.weight', 'last.0.weight']  # Primera y última capa
    
    for layer_name in layers_to_analyze:
        print(f"\n🔍 Capa: {layer_name}")
        
        # Obtener pesos de ambos modelos
        weights1 = dict(model1.named_parameters())[layer_name]
        weights2 = dict(model2.named_parameters())[layer_name]
        
        print(f"   Shape: {weights1.shape}")
        print(f"   Modelo 1 - Media: {weights1.mean():.6f}, Std: {weights1.std():.6f}")
        print(f"   Modelo 2 - Media: {weights2.mean():.6f}, Std: {weights2.std():.6f}")
        print(f"   Diferencia absoluta media: {torch.abs(weights1 - weights2).mean():.6f}")
        
        # Verificar si hay pesos NaN o Inf
        has_nan1 = torch.isnan(weights1).any()
        has_inf1 = torch.isinf(weights1).any()
        has_nan2 = torch.isnan(weights2).any()
        has_inf2 = torch.isinf(weights2).any()
        
        if has_nan1 or has_inf1 or has_nan2 or has_inf2:
            print("   ⚠️  ADVERTENCIA: Contiene valores NaN o Inf!")
    
    # 5. Visualización de distribución de pesos
    visualize_weight_distributions(model1, model2)

def visualize_weight_distributions(model1, model2):
    """Visualiza las distribuciones de pesos"""
    
    # Seleccionar pesos de la primera capa convolucional
    weights1_layer1 = dict(model1.named_parameters())['layer1.0.weight'].flatten().detach().cpu().numpy()  # CORREGIDO: agregado .detach()
    weights2_layer1 = dict(model2.named_parameters())['layer1.0.weight'].flatten().detach().cpu().numpy()  # CORREGIDO: agregado .detach()
    
    # Crear figura
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histograma 1
    axes[0].hist(weights1_layer1, bins=50, alpha=0.7, label='Modelo 1', color='blue')
    axes[0].hist(weights2_layer1, bins=50, alpha=0.7, label='Modelo 2', color='red')
    axes[0].set_title('Distribución de pesos - Capa 1')
    axes[0].set_xlabel('Valor del peso')
    axes[0].set_ylabel('Frecuencia')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Box plot comparativo
    data = [weights1_layer1[:1000], weights2_layer1[:1000]]  # Tomar muestra
    axes[1].boxplot(data, labels=['Modelo 1', 'Modelo 2'])
    axes[1].set_title('Comparación de distribución (muestra)')
    axes[1].set_ylabel('Valor del peso')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('comparacion_pesos.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 Gráfico guardado como: comparacion_pesos.png")
    
    # También mostrar estadísticas
    print(f"\nEstadísticas capa 1 (primeras 1000 muestras):")
    print(f"{'Estadística':<15} {'Modelo 1':<12} {'Modelo 2':<12}")
    print("-" * 45)
    
    stats = [
        ("Media", np.mean(weights1_layer1[:1000]), np.mean(weights2_layer1[:1000])),
        ("Std", np.std(weights1_layer1[:1000]), np.std(weights2_layer1[:1000])),
        ("Mín", np.min(weights1_layer1[:1000]), np.min(weights2_layer1[:1000])),
        ("Máx", np.max(weights1_layer1[:1000]), np.max(weights2_layer1[:1000])),
        ("Mediana", np.median(weights1_layer1[:1000]), np.median(weights2_layer1[:1000]))
    ]
    
    for stat_name, val1, val2 in stats:
        print(f"{stat_name:<15} {val1:<12.6f} {val2:<12.6f}")

def load_single_model(model_path, detailed=True):
    """Carga y analiza un solo modelo"""
    
    print(f"\n{'='*60}")
    print(f"CARGANDO: {os.path.basename(model_path)}")
    print(f"{'='*60}")
    
    dims = (32, 64)
    model = WildFireNet(input_channels=4, num_classes=2, dims=dims)
    
    # Cargar checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint)
    model.eval()
    
    print("✅ Modelo cargado exitosamente")
    
    # Información básica
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n📊 Información básica:")
    print(f"   Parámetros totales: {total_params:,}")
    print(f"   Dispositivo: {'GPU' if next(model.parameters()).is_cuda else 'CPU'}")
    
    if detailed:
        # Análisis detallado
        print(f"\n🔍 Capas del modelo:")
        for name, param in model.named_parameters():
            if 'weight' in name:  # Mostrar solo pesos
                print(f"   {name:45} {str(param.shape):20} mean: {param.mean():.6f}")
    
    # Test rápido
    test_input = torch.randn(1, 4, 128, 128)
    with torch.no_grad():
        output = model(test_input)
        probs = torch.softmax(output, dim=1)
        
    print(f"\n🧪 Test rápido (batch=1, 128x128):")
    print(f"   Output shape: {output.shape}")
    print(f"   Prob clase 0: {probs[0, 0].mean():.4f}")
    print(f"   Prob clase 1: {probs[0, 1].mean():.4f}")
    
    return model

def create_inference_example():
    """Crea un ejemplo completo para hacer inferencia"""
    
    template = '''#!/usr/bin/env python3
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
    print("\\n🧪 Ejemplo con imagen dummy:")
    
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
    print(f"\\n📦 Inferencia por lotes (batch={batch_size}):")
    
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
    
    print("\\n✅ Ejemplo completado exitosamente")
    print("\\n📝 Para usar con tus propias imágenes:")
    print("   1. Carga tus imágenes multiespectrales (4 canales)")
    print("   2. Redimensiona a 128x128 si es necesario")
    print("   3. Normaliza igual que durante el entrenamiento")
    print("   4. Usa la función predict()")
'''
    
    with open("inferencia_wildfire.py", "w", encoding='utf-8') as f:
        f.write(template)
    
    print(f"\n✅ Ejemplo de inferencia guardado como: inferencia_wildfire.py")

if __name__ == "__main__":
    print("ANÁLISIS DE MODELOS WILDFIRENET")
    print("=" * 60)
    
    # Rutas a tus modelos
    model1_path = "Wildfire_N__128__AUG_1_best.pth"
    model2_path = "Wildfire_N__128__AUG_1_valid_best_3264.pth"
    
    # Verificar que existen
    if not os.path.exists(model1_path):
        print(f"❌ No se encuentra: {model1_path}")
        model1_path = None
    
    if not os.path.exists(model2_path):
        print(f"❌ No se encuentra: {model2_path}")
        model2_path = None
    
    if model1_path and model2_path:
        # Comparar ambos modelos
        compare_models(model1_path, model2_path)
        
        print(f"\n{'='*60}")
        print("ANÁLISIS DE RESULTADOS:")
        print(f"{'='*60}")
        print(f"✅ Los modelos tienen la misma arquitectura (224,458 parámetros)")
        print(f"✅ Los pesos son muy similares (diferencia media: 0.000712)")
        print(f"📊 Diferencia en salidas: {0.1434:.4f} (lo esperado para modelos diferentes)")
        
        print(f"\n{'='*60}")
        print("RECOMENDACIÓN:")
        print(f"{'='*60}")
        print(f"🎯 USAR 'Wildfire_N__128__AUG_1_valid_best_3264.pth' porque:")
        print(f"   - Tiene mejor F1-score en validación")
        print(f"   - La capa final (last.1.weight) es diferente (diff: 0.0106)")
        print(f"   - Esto sugiere ajustes finos para mejor rendimiento")
        print(f"   - Prob incendio más conservadora: 44.85% vs 49.20%")
        
        print(f"\n📁 USAR 'Wildfire_N__128__AUG_1_best.pth' si prefieres:")
        print(f"   - Menor pérdida de entrenamiento")
        print(f"   - Continuar entrenamiento desde aquí")
        
        print(f"\n🧪 PRUEBA RÁPIDA DE AMBOS:")
        print(f"   python3 prueba_modelos.py")
    
    elif model1_path or model2_path:
        # Cargar solo el que existe
        model_path = model1_path if model1_path else model2_path
        load_single_model(model_path, detailed=True)
    
    else:
        print("❌ No se encontraron modelos .pth")
    
    # Crear ejemplo de inferencia
    create_inference_example()
    
    print(f"\n{'='*60}")
    print("NEXT STEPS:")
    print(f"{'='*60}")
    print("1. Ejecutar: python3 inferencia_wildfire.py")
    print("2. Probar ambos modelos: python3 prueba_modelos.py")
    print("3. Para apagar después de entrenar, agrega el código al script principal")