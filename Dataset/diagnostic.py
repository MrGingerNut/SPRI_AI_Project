#!/usr/bin/env python3
"""
diagnostico_tiff.py - Analiza archivos TIFF para ver su contenido real
"""

import numpy as np
from osgeo import gdal
import sys
import matplotlib.pyplot as plt

def analizar_tiff(ruta):
    """Analiza en detalle un archivo TIFF"""
    print(f"\n{'='*60}")
    print(f"ANÁLISIS DE: {ruta}")
    print(f"{'='*60}")
    
    ds = gdal.Open(ruta)
    if ds is None:
        print(f"ERROR: No se puede abrir {ruta}")
        return
    
    # Información básica
    print(f"Dimensiones: {ds.RasterXSize} x {ds.RasterYSize}")
    print(f"Número de bandas: {ds.RasterCount}")
    
    for i in range(1, ds.RasterCount + 1):
        banda = ds.GetRasterBand(i)
        arr = banda.ReadAsArray()
        
        print(f"\n--- Banda {i} ---")
        print(f"  Tipo de datos: {gdal.GetDataTypeName(banda.DataType)}")
        print(f"  Tamaño: {arr.shape}")
        print(f"  Valores únicos: {np.unique(arr)}")
        print(f"  Mínimo: {np.min(arr)}")
        print(f"  Máximo: {np.max(arr)}")
        print(f"  Media: {np.mean(arr):.2f}")
        print(f"  Desviación estándar: {np.std(arr):.2f}")
        
        # Contar valores específicos
        total = arr.size
        ceros = np.sum(arr == 0)
        blancos = np.sum(arr == 255)
        otros = total - ceros - blancos
        
        print(f"  Píxeles con valor 0: {ceros} ({ceros/total*100:.2f}%)")
        print(f"  Píxeles con valor 255: {blancos} ({blancos/total*100:.2f}%)")
        print(f"  Píxeles con otros valores: {otros} ({otros/total*100:.2f}%)")
    
    # Intentar crear una visualización
    if ds.RasterCount >= 3:
        print(f"\nCreando visualización RGB...")
        
        # Leer las 3 primeras bandas
        r = ds.GetRasterBand(1).ReadAsArray()
        g = ds.GetRasterBand(2).ReadAsArray()
        b = ds.GetRasterBand(3).ReadAsArray()
        
        # Combinar en una imagen RGB
        rgb = np.stack([r, g, b], axis=-1)
        
        # Mostrar
        plt.figure(figsize=(12, 8))
        plt.imshow(rgb)
        plt.title(f"Visualización de {ruta}")
        plt.axis('off')
        plt.colorbar(label='Valor del píxel')
        plt.show()
    
    ds = None

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Uso: python diagnostico_tiff.py <ruta_al_tiff>")
        sys.exit(1)
    
    analizar_tiff(sys.argv[1])
