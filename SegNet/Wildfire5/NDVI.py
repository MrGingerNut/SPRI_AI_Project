import rasterio
import numpy as np
import matplotlib.pyplot as plt

# Rutas
archivo = "/home/liese2/SPRI_AI_project/Inferencias/Burned/Huamuchil_Oaxaca_Burned.tiff"
output_path = "/home/liese2/SPRI_AI_project/Inferencias/Burned/NDVI_Huamuchil.tiff"

# Abrir el archivo
with rasterio.open(archivo) as src:
    # Leer las bandas
    banda1 = src.read(1).astype(float)  # Red (Rojo)
    banda2 = src.read(2).astype(float)  # NIR (Infrarrojo cercano)
    banda3 = src.read(3).astype(float)  # Green (Verde) - no se usa para NDVI
    
    profile = src.profile
    
    print("Configuración de bandas:")
    print(f"  Banda 1 (Red): min={banda1.min():.0f}, max={banda1.max():.0f}, media={banda1.mean():.0f}")
    print(f"  Banda 2 (NIR): min={banda2.min():.0f}, max={banda2.max():.0f}, media={banda2.mean():.0f}")
    print(f"  Banda 3 (Green): min={banda3.min():.0f}, max={banda3.max():.0f}, media={banda3.mean():.0f}")

# Calcular NDVI con la configuración correcta (B1=Red, B2=NIR)
np.seterr(divide='ignore', invalid='ignore')
ndvi = (banda2 - banda1) / (banda2 + banda1)

# Manejar división por cero (donde NIR + Red = 0)
ndvi = np.where((banda2 + banda1) == 0, -1, ndvi)

print(f"\nEstadísticas NDVI final:")
print(f"  Mínimo: {ndvi.min():.3f}")
print(f"  Máximo: {ndvi.max():.3f}")
print(f"  Promedio: {ndvi.mean():.3f}")
print(f"  Desviación estándar: {ndvi.std():.3f}")

# Guardar el NDVI como GeoTIFF
profile.update(
    dtype=rasterio.float32,
    count=1,
    compress='lzw',
    nodata=-999
)

with rasterio.open(output_path, 'w', **profile) as dst:
    dst.write(ndvi.astype(rasterio.float32), 1)

print(f"\n✓ NDVI guardado en: {output_path}")

# Visualización con mejor resolución
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Imagen original (falso color para visualización)
# Normalizar para visualización
norm_banda1 = np.clip(banda1 / banda1.max(), 0, 1)
norm_banda2 = np.clip(banda2 / banda2.max(), 0, 1)
norm_banda3 = np.clip(banda3 / banda3.max(), 0, 1)

# Mostrar composición falso color (NIR, Red, Green)
falso_color = np.stack([norm_banda2, norm_banda1, norm_banda3], axis=2)
axes[0].imshow(falso_color)
axes[0].set_title('Imagen en Falso Color\n(NIR = Rojo, Red = Verde, Green = Azul)', fontsize=12)
axes[0].axis('off')

# Mostrar NDVI
im = axes[1].imshow(ndvi, cmap='RdYlGn', vmin=-1, vmax=1)
axes[1].set_title('NDVI - Zona de Incendio\n(Valores negativos = área quemada)', fontsize=12)
axes[1].axis('off')
plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label='NDVI')

plt.tight_layout()
plt.savefig('/home/liese2/SPRI_AI_project/Inferencias/Burned/NDVI_Huamuchil_visualizacion.png', dpi=150, bbox_inches='tight')
plt.show()

# Análisis adicional: Clasificar áreas quemadas
print("\n" + "="*50)
print("ANÁLISIS DE ÁREAS QUEMADAS:")
print("="*50)

# Definir umbrales para áreas quemadas (ajusta según necesidad)
quemado_severo = (ndvi < -0.3)
quemado_moderado = (ndvi >= -0.3) & (ndvi < 0)
vegetacion_afectada = (ndvi >= 0) & (ndvi < 0.2)
vegetacion_sana = (ndvi >= 0.2)

porcentaje_quemado_severo = np.sum(quemado_severo) / ndvi.size * 100
porcentaje_quemado_moderado = np.sum(quemado_moderado) / ndvi.size * 100
porcentaje_vegetacion_afectada = np.sum(vegetacion_afectada) / ndvi.size * 100
porcentaje_vegetacion_sana = np.sum(vegetacion_sana) / ndvi.size * 100

print(f"Área quemada severa (NDVI < -0.3): {porcentaje_quemado_severo:.1f}%")
print(f"Área quemada moderada (-0.3 ≤ NDVI < 0): {porcentaje_quemado_moderado:.1f}%")
print(f"Vegetación afectada (0 ≤ NDVI < 0.2): {porcentaje_vegetacion_afectada:.1f}%")
print(f"Vegetación sana (NDVI ≥ 0.2): {porcentaje_vegetacion_sana:.1f}%")
print(f"\nTotal área afectada por fuego: {porcentaje_quemado_severo + porcentaje_quemado_moderado:.1f}%")