#!/usr/bin/env python3
# Conversion.py - Convertir Wildfire3_N__128__AUG_1_best.pth a formato binario

import torch
import numpy as np
import os
import sys
import json
import struct

def convert_wildfire_weights(pth_file="Wildfire3_N__128__AUG_1_best.pth", output_dir="wildfire_bin"):
    """Convierte pesos de WildFireNet PyTorch a binario para C/embebidos"""
    
    print("=" * 60)
    print(f"CONVERSIÓN DE PESOS WildFireNet")
    print(f"Archivo de entrada: {pth_file}")
    print(f"Directorio de salida: {output_dir}")
    print("=" * 60)
    
    # Verificar que el archivo existe
    if not os.path.exists(pth_file):
        print(f"❌ ERROR: No se encuentra el archivo {pth_file}")
        print(f"   Buscando en: {os.path.abspath('.')}")
        return False
    
    print(f"📥 Cargando checkpoint desde {pth_file}...")
    
    try:
        # Cargar checkpoint
        checkpoint = torch.load(pth_file, map_location='cpu', weights_only=True)
        print("✓ Checkpoint cargado correctamente")
    except Exception as e:
        print(f"❌ Error al cargar el checkpoint: {e}")
        return False
    
    # Extraer state_dict
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        print("✓ Se encontró 'state_dict' en el checkpoint")
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print("✓ Se encontró 'model_state_dict' en el checkpoint")
    else:
        state_dict = checkpoint
        print("✓ Usando pesos directamente (sin wrapper)")
    
    # Filtrar solo parámetros (no buffers)
    param_dict = {k: v for k, v in state_dict.items() if v.dtype == torch.float32}
    print(f"✓ {len(param_dict)} parámetros encontrados (float32)")
    
    # Crear directorio de salida
    os.makedirs(output_dir, exist_ok=True)
    
    # Metadatos del modelo
    metadata = {
        'model_name': 'Wildfire3_N_128_AUG_1',
        'original_file': pth_file,
        'conversion_date': np.datetime64('now').astype(str),
        'layers': [],
        'total_parameters': 0,
        'total_size_mb': 0
    }
    
    print("\n📊 PROCESANDO CAPAS:")
    print("-" * 80)
    
    # Procesar cada capa
    for i, (name, param) in enumerate(param_dict.items()):
        # Crear nombre seguro para archivos
        safe_name = name.replace('.', '_').replace(':', '_')
        safe_name = safe_name.replace('module.', '').replace('model.', '')
        
        # Convertir a numpy
        param_np = param.numpy().astype(np.float32)
        
        # Estadísticas de la capa
        layer_info = {
            'id': i,
            'original_name': name,
            'safe_name': safe_name,
            'shape': list(param_np.shape),
            'dtype': 'float32',
            'size_bytes': int(param_np.nbytes),
            'size_mb': round(param_np.nbytes / (1024*1024), 3),
            'mean': float(np.mean(param_np)),
            'std': float(np.std(param_np)),
            'min': float(np.min(param_np)),
            'max': float(np.max(param_np))
        }
        
        metadata['layers'].append(layer_info)
        metadata['total_parameters'] += param_np.size
        metadata['total_size_mb'] += param_np.nbytes / (1024*1024)
        
        # Guardar como archivo binario individual
        bin_filename = f"layer_{i:03d}_{safe_name}.bin"
        bin_path = os.path.join(output_dir, bin_filename)
        param_np.tofile(bin_path)
        
        # Mostrar progreso
        shape_str = 'x'.join(map(str, param_np.shape))
        print(f"  [{i+1:03d}/{len(param_dict):03d}] {name[:50]:50}")
        print(f"       → {bin_filename:30} [{shape_str:15}] {param_np.nbytes/1024:7.1f} KB")
    
    # Estadísticas de normalización (ajustar según tu modelo)
    print("\n📈 GUARDANDO ESTADÍSTICAS DE NORMALIZACIÓN...")
    
    # Valores de ejemplo - REEMPLAZAR con los valores reales de tu modelo
    input_mean = np.array([0.1527, 0.1216, 0.1118, 0.2149], dtype=np.float32)
    input_std = np.array([0.0618, 0.0497, 0.0428, 0.0921], dtype=np.float32)
    
    input_mean.tofile(os.path.join(output_dir, 'norm_mean.bin'))
    input_std.tofile(os.path.join(output_dir, 'norm_std.bin'))
    
    metadata['normalization'] = {
        'mean': input_mean.tolist(),
        'std': input_std.tolist()
    }
    
    # Guardar metadatos
    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    # Crear archivo de resumen
    summary_path = os.path.join(output_dir, 'SUMMARY.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("RESUMEN DE CONVERSIÓN WILDFIRENET\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Modelo original: {pth_file}\n")
        f.write(f"Fecha conversión: {metadata['conversion_date']}\n")
        f.write(f"Total capas: {len(param_dict)}\n")
        f.write(f"Total parámetros: {metadata['total_parameters']:,}\n")
        f.write(f"Tamaño total: {metadata['total_size_mb']:.2f} MB\n\n")
        
        f.write("LISTA DE CAPAS:\n")
        f.write("-" * 80 + "\n")
        for layer in metadata['layers']:
            shape_str = 'x'.join(map(str, layer['shape']))
            f.write(f"[{layer['id']:03d}] {layer['safe_name'][:40]:40} ")
            f.write(f"{shape_str:15} {layer['size_mb']:6.2f} MB\n")
    
    # Crear archivo de cabecera C simple
    header_path = os.path.join(output_dir, 'wildfire_weights.h')
    with open(header_path, 'w') as f:
        f.write("""#ifndef WILDFIRE_WEIGHTS_H
#define WILDFIRE_WEIGHTS_H

// Estadísticas de normalización
extern const float wildfire_norm_mean[4];
extern const float wildfire_norm_std[4];

// Número total de capas
#define WILDFIRE_NUM_LAYERS %d

// Estructura para información de capas
typedef struct {
    const char* name;
    const float* data;
    int shape[4];
    int ndim;
    int size;
} WildfireLayer;

// Declaraciones externas
#ifdef __cplusplus
extern "C" {
#endif

// Cargar pesos
int wildfire_load_weights(const char* base_path);

// Obtener capa por índice
const float* wildfire_get_layer_data(int layer_index, int* out_size);

// Obtener capa por nombre
const float* wildfire_get_layer_by_name(const char* name, int* out_size);

// Liberar memoria
void wildfire_free_weights(void);

#ifdef __cplusplus
}
#endif

#endif // WILDFIRE_WEIGHTS_H
""" % len(param_dict))
    
    print("\n" + "=" * 60)
    print("✅ CONVERSIÓN COMPLETADA EXITOSAMENTE")
    print("=" * 60)
    print(f"📂 Directorio de salida: {os.path.abspath(output_dir)}")
    print(f"📊 Total capas procesadas: {len(param_dict)}")
    print(f"🧮 Total parámetros: {metadata['total_parameters']:,}")
    print(f"💾 Tamaño total: {metadata['total_size_mb']:.2f} MB")
    print(f"📝 Metadatos: metadata.json")
    print(f"📋 Resumen: SUMMARY.txt")
    print(f"🔧 Cabecera C: wildfire_weights.h")
    print(f"📊 Estadísticas: norm_mean.bin, norm_std.bin")
    print("\n📁 Archivos generados:")
    
    # Listar archivos generados
    for file in sorted(os.listdir(output_dir)):
        if file.endswith('.bin'):
            size = os.path.getsize(os.path.join(output_dir, file))
            print(f"  • {file:30} - {size/1024:7.1f} KB")
    
    return True

def verificar_conversion(output_dir):
    """Verifica la integridad de la conversión"""
    print("\n" + "=" * 60)
    print("🔍 VERIFICANDO CONVERSIÓN")
    print("=" * 60)
    
    if not os.path.exists(output_dir):
        print("❌ Directorio no existe")
        return False
    
    # Contar archivos .bin
    bin_files = [f for f in os.listdir(output_dir) if f.endswith('.bin')]
    print(f"✓ Archivos .bin encontrados: {len(bin_files)}")
    
    # Verificar metadatos
    metadata_path = os.path.join(output_dir, 'metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"✓ Metadatos válidos: {metadata['model_name']}")
        print(f"✓ Capas en metadatos: {len(metadata['layers'])}")
    
    # Verificar archivos de normalización
    if os.path.exists(os.path.join(output_dir, 'norm_mean.bin')):
        mean_data = np.fromfile(os.path.join(output_dir, 'norm_mean.bin'), dtype=np.float32)
        print(f"✓ Estadísticas mean: {mean_data.shape[0]} valores")
    
    print("✅ Verificación completada")
    return True

if __name__ == "__main__":
    # Configuración por defecto
    ARCHIVO_PTH = "Wildfire3_N__128__AUG_1_best.pth"
    DIRECTORIO_SALIDA = "wildfire_bin"
    
    # Usar argumentos de línea de comandos si se proporcionan
    if len(sys.argv) > 1:
        ARCHIVO_PTH = sys.argv[1]
    if len(sys.argv) > 2:
        DIRECTORIO_SALIDA = sys.argv[2]
    
    # Ejecutar conversión
    if convert_wildfire_weights(ARCHIVO_PTH, DIRECTORIO_SALIDA):
        verificar_conversion(DIRECTORIO_SALIDA)
        
        print("\n" + "=" * 60)
        print("🎯 USO EN C/C++:")
        print("=" * 60)
        print("1. Copia los archivos .bin a tu proyecto")
        print("2. Incluye 'wildfire_weights.h'")
        print("3. Implementa funciones de carga:")
        print("""
   // Ejemplo:
   float* load_binary(const char* filename, size_t* size) {
       FILE* f = fopen(filename, "rb");
       fseek(f, 0, SEEK_END);
       *size = ftell(f) / sizeof(float);
       rewind(f);
       float* data = (float*)malloc(*size * sizeof(float));
       fread(data, sizeof(float), *size, f);
       fclose(f);
       return data;
   }
        """)
        print("4. Los pesos están en formato float32 plano")