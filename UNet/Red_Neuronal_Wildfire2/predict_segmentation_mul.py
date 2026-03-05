#!/usr/bin/env python3
"""
Script para realizar predicciones en múltiples imágenes de forma automática
"""

import os
import glob
import argparse
from predict_wildfire import predict_image

def batch_predict(model_path, input_dir, output_dir, file_pattern="*.tif", 
                 gpu="auto", threshold=0.5, visualization=False):
    """
    Realizar predicciones en múltiples imágenes
    
    Args:
        model_path: Ruta al modelo entrenado
        input_dir: Directorio con imágenes de entrada
        output_dir: Directorio para guardar predicciones
        file_pattern: Patrón para filtrar archivos (ej: "*.tif", "imagen_*.tif")
        gpu: Usar GPU (True/False/"auto")
        threshold: Umbral de probabilidad
        visualization: Generar visualizaciones
    """
    
    print("="*60)
    print("PREDICCIÓN EN BATCH - INCENDIOS FORESTALES")
    print("="*60)
    print(f"Modelo: {model_path}")
    print(f"Directorio de entrada: {input_dir}")
    print(f"Directorio de salida: {output_dir}")
    print(f"Patrón de archivos: {file_pattern}")
    print(f"Umbral: {threshold}")
    print("="*60 + "\n")
    
    # Verificar que existe el modelo
    if not os.path.exists(model_path):
        print(f"ERROR: No se encontró el modelo en {model_path}")
        return
    
    # Verificar que existe el directorio de entrada
    if not os.path.exists(input_dir):
        print(f"ERROR: No existe el directorio {input_dir}")
        return
    
    # Crear directorio de salida si no existe
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"✓ Directorio de salida creado: {output_dir}\n")
    
    # Buscar imágenes
    search_path = os.path.join(input_dir, file_pattern)
    image_files = glob.glob(search_path)
    
    if len(image_files) == 0:
        print(f"ERROR: No se encontraron imágenes con el patrón '{file_pattern}' en {input_dir}")
        return
    
    print(f"Se encontraron {len(image_files)} imágenes para procesar\n")
    
    # Procesar cada imagen
    success_count = 0
    error_count = 0
    
    for i, image_path in enumerate(image_files, 1):
        print(f"\n{'='*60}")
        print(f"PROCESANDO IMAGEN {i}/{len(image_files)}")
        print(f"{'='*60}")
        
        # Obtener nombre de archivo sin extensión
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Crear ruta de salida
        output_path = os.path.join(output_dir, f"{base_name}_prediction.tif")
        
        try:
            # Realizar predicción
            predict_image(
                model_path=model_path,
                image_path=image_path,
                output_path=output_path,
                gpu=gpu,
                threshold=threshold,
                visualization=visualization
            )
            success_count += 1
            
        except Exception as e:
            print(f"\n❌ ERROR procesando {image_path}:")
            print(f"   {str(e)}\n")
            error_count += 1
    
    # Resumen final
    print("\n" + "="*60)
    print("RESUMEN DE PROCESAMIENTO EN BATCH")
    print("="*60)
    print(f"Total de imágenes: {len(image_files)}")
    print(f"Exitosas: {success_count}")
    print(f"Con errores: {error_count}")
    print(f"Predicciones guardadas en: {output_dir}")
    print("="*60)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predicción en Batch de Incendios Forestales')
    parser.add_argument('--model', required=True, help='Ruta al modelo entrenado (.pth)')
    parser.add_argument('--input-dir', required=True, help='Directorio con imágenes de entrada')
    parser.add_argument('--output-dir', required=True, help='Directorio para guardar predicciones')
    parser.add_argument('--pattern', default="*.tif", help='Patrón de archivos a procesar (default: *.tif)')
    parser.add_argument('--gpu', type=str, default="auto", 
                       help='Usar GPU: True/False/auto (auto detecta disponibilidad)')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Umbral de probabilidad para clasificar como incendio (0-1)')
    parser.add_argument('--visualization', type=str, default="False",
                       help='Generar visualizaciones adicionales')
    
    args = parser.parse_args()
    
    batch_predict(
        model_path=args.model,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        file_pattern=args.pattern,
        gpu=args.gpu,
        threshold=args.threshold,
        visualization=args.visualization.lower() == "true"
    )