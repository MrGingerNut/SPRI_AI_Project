#!/usr/bin/env python3
"""
convert_binary_to_visible.py - Convierte predicciones binarias (0/1) a imágenes visibles
"""

import numpy as np
from osgeo import gdal
import sys
import os
import cv2

def convertir_tiff_a_visible(ruta_entrada, ruta_salida=None, color_incendio='blanco'):
    """
    Convierte un TIFF binario (0/1) a una imagen visible.
    
    Args:
        ruta_entrada: Ruta al TIFF binario
        ruta_salida: Ruta para guardar (si None, se agrega '_visible')
        color_incendio: 'blanco', 'rojo', o 'amarillo'
    """
    
    print(f"\n{'='*60}")
    print(f"CONVIRTIENDO TIFF BINARIO A VISIBLE")
    print(f"{'='*60}")
    print(f"Entrada: {ruta_entrada}")
    
    # Cargar el TIFF
    ds = gdal.Open(ruta_entrada)
    if ds is None:
        print(f"ERROR: No se puede abrir {ruta_entrada}")
        return
    
    # Leer datos
    banda = ds.GetRasterBand(1)
    datos = banda.ReadAsArray()
    height, width = datos.shape
    
    print(f"Dimensiones: {width} x {height}")
    print(f"Valores únicos: {np.unique(datos)}")
    print(f"Píxeles con valor 1 (incendios): {np.sum(datos == 1)}")
    print(f"Píxeles con valor 0 (fondo): {np.sum(datos == 0)}")
    
    # Crear imagen RGB visible
    rgb_visible = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Obtener máscara de incendios
    mascara_incendios = datos == 1
    
    # Aplicar color según opción
    if color_incendio.lower() == 'blanco':
        rgb_visible[mascara_incendios] = [255, 255, 255]  # Blanco
        print(f"✓ Incendios coloreados en BLANCO")
    elif color_incendio.lower() == 'rojo':
        rgb_visible[mascara_incendios] = [255, 0, 0]      # Rojo
        print(f"✓ Incendios coloreados en ROJO")
    elif color_incendio.lower() == 'amarillo':
        rgb_visible[mascara_incendios] = [255, 255, 0]    # Amarillo
        print(f"✓ Incendios coloreados en AMARILLO")
    else:
        rgb_visible[mascara_incendios] = [255, 255, 255]  # Blanco por defecto
        print(f"✓ Incendios coloreados en BLANCO (por defecto)")
    
    # Agregar marcas de verificación en las esquinas
    # Esquina superior izquierda: Verde
    rgb_visible[0:5, 0:5] = [0, 255, 0]
    # Esquina superior derecha: Azul
    rgb_visible[0:5, -5:] = [0, 0, 255]
    # Esquina inferior izquierda: Cian
    rgb_visible[-5:, 0:5] = [0, 255, 255]
    # Esquina inferior derecha: Magenta
    rgb_visible[-5:, -5:] = [255, 0, 255]
    
    print(f"✓ Marcas de verificación agregadas en las esquinas")
    
    # Determinar ruta de salida
    if ruta_salida is None:
        base, ext = os.path.splitext(ruta_entrada)
        ruta_salida = f"{base}_visible.png"
    
    # Determinar formato por extensión
    ext = os.path.splitext(ruta_salida)[1].lower()
    
    if ext in ['.png', '.jpg', '.jpeg']:
        # Guardar como PNG/JPG
        cv2.imwrite(ruta_salida, cv2.cvtColor(rgb_visible, cv2.COLOR_RGB2BGR))
        print(f"✓ Imagen visible guardada como: {ruta_salida}")
        
    elif ext in ['.tif', '.tiff']:
        # Guardar como GeoTIFF RGB
        driver = gdal.GetDriverByName('GTiff')
        out_ds = driver.Create(ruta_salida, width, height, 3, gdal.GDT_Byte)
        
        # Copiar georreferencia
        out_ds.SetGeoTransform(ds.GetGeoTransform())
        out_ds.SetProjection(ds.GetProjection())
        
        # Guardar canales RGB
        out_ds.GetRasterBand(1).WriteArray(rgb_visible[:, :, 0])
        out_ds.GetRasterBand(2).WriteArray(rgb_visible[:, :, 1])
        out_ds.GetRasterBand(3).WriteArray(rgb_visible[:, :, 2])
        
        # Configurar para visualización
        out_ds.GetRasterBand(1).SetColorInterpretation(gdal.GCI_RedBand)
        out_ds.GetRasterBand(2).SetColorInterpretation(gdal.GCI_GreenBand)
        out_ds.GetRasterBand(3).SetColorInterpretation(gdal.GCI_BlueBand)
        
        out_ds.FlushCache()
        out_ds = None
        print(f"✓ Imagen visible guardada como GeoTIFF: {ruta_salida}")
        
    else:
        # Por defecto, guardar como PNG
        ruta_salida = os.path.splitext(ruta_salida)[0] + '.png'
        cv2.imwrite(ruta_salida, cv2.cvtColor(rgb_visible, cv2.COLOR_RGB2BGR))
        print(f"✓ Imagen visible guardada como PNG: {ruta_salida}")
    
    # También crear una versión de diagnóstico
    crear_diagnostico(datos, rgb_visible, ruta_entrada)
    
    ds = None
    print(f"\n{'='*60}")
    print(f"CONVERSIÓN COMPLETADA")
    print(f"{'='*60}")

def crear_diagnostico(datos_binarios, datos_rgb, ruta_original):
    """Crea una imagen de diagnóstico"""
    
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Imagen binaria original
    ax1 = axes[0]
    im1 = ax1.imshow(datos_binarios, cmap='gray', vmin=0, vmax=1)
    ax1.set_title(f'Imagen Binaria Original\n0s: {np.sum(datos_binarios==0):,} | 1s: {np.sum(datos_binarios==1):,}')
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    # Imagen RGB visible
    ax2 = axes[1]
    im2 = ax2.imshow(cv2.cvtColor(datos_rgb, cv2.COLOR_RGB2BGR))
    ax2.set_title('Imagen Convertida (Visible)')
    ax2.axis('off')
    
    # Guardar diagnóstico
    base = os.path.splitext(ruta_original)[0]
    ruta_diagnostico = f"{base}_diagnostico.png"
    plt.tight_layout()
    plt.savefig(ruta_diagnostico, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Diagnóstico guardado como: {ruta_diagnostico}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Uso: python convert_binary_to_visible.py <ruta_tiff> [ruta_salida] [color]")
        print("  color: blanco, rojo, amarillo (default: blanco)")
        sys.exit(1)
    
    ruta_entrada = sys.argv[1]
    
    ruta_salida = sys.argv[2] if len(sys.argv) > 2 else None
    color = sys.argv[3] if len(sys.argv) > 3 else 'blanco'
    
    convertir_tiff_a_visible(ruta_entrada, ruta_salida, color)
