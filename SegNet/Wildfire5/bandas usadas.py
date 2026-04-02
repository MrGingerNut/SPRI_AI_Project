import rasterio
import numpy as np
import matplotlib.pyplot as plt

archivo = "/home/liese2/SPRI_AI_project/Inferencias/Burned/Huamuchil_Oaxaca_Burned.tiff"

with rasterio.open(archivo) as src:
    banda1 = src.read(1).astype(float)
    banda2 = src.read(2).astype(float)
    banda3 = src.read(3).astype(float)
    
    print("="*60)
    print("VERIFICACIÓN DE BANDAS")
    print("="*60)
    
    # Calcular relaciones típicas
    relacion_12 = banda1 / (banda2 + 0.01)  # Relación Banda1/Banda2
    relacion_21 = banda2 / (banda1 + 0.01)  # Relación Banda2/Banda1
    
    print(f"\nBanda 1 - Rango: [{banda1.min():.0f}, {banda1.max():.0f}] - Media: {banda1.mean():.0f}")
    print(f"Banda 2 - Rango: [{banda2.min():.0f}, {banda2.max():.0f}] - Media: {banda2.mean():.0f}")
    print(f"Banda 3 - Rango: [{banda3.min():.0f}, {banda3.max():.0f}] - Media: {banda3.mean():.0f}")
    
    # Identificación por valores típicos
    print("\n" + "="*60)
    print("ANÁLISIS DE COMPORTAMIENTO ESPECTRAL")
    print("="*60)
    
    # En vegetación sana: NIR > Red
    # En zonas quemadas: Red > NIR (pero ambos bajos)
    
    # Calcular proporción de píxeles donde Banda1 > Banda2
    prop_b1_mayor_b2 = np.sum(banda1 > banda2) / banda1.size * 100
    prop_b2_mayor_b1 = np.sum(banda2 > banda1) / banda1.size * 100
    
    print(f"\nPíxeles donde Banda1 > Banda2: {prop_b1_mayor_b2:.1f}%")
    print(f"Píxeles donde Banda2 > Banda1: {prop_b2_mayor_b1:.1f}%")
    
    # Si es una zona quemada, debería haber más píxeles donde Red > NIR
    # porque la vegetación muerta refleja menos NIR
    
    if prop_b1_mayor_b2 > 50:
        print("\n🔍 Posible configuración: Banda1 = Red, Banda2 = NIR")
        print("   (Red tiene valores más altos que NIR - típico en zonas quemadas)")
        ndvi_test = (banda2 - banda1) / (banda2 + banda1)
        config = "B1=Red, B2=NIR"
    else:
        print("\n🔍 Posible configuración: Banda1 = NIR, Banda2 = Red")
        print("   (NIR tiene valores más altos que Red - típico en vegetación sana)")
        ndvi_test = (banda1 - banda2) / (banda1 + banda2)
        config = "B1=NIR, B2=Red"
    
    print(f"\nNDVI con {config}:")
    print(f"  Mínimo: {ndvi_test.min():.3f}")
    print(f"  Máximo: {ndvi_test.max():.3f}")
    print(f"  Media: {ndvi_test.mean():.3f}")
    
    # Visualizar para confirmar visualmente
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Normalizar para visualización
    norm1 = np.clip((banda1 - banda1.min()) / (banda1.max() - banda1.min()), 0, 1)
    norm2 = np.clip((banda2 - banda2.min()) / (banda2.max() - banda2.min()), 0, 1)
    norm3 = np.clip((banda3 - banda3.min()) / (banda3.max() - banda3.min()), 0, 1)
    
    # Opción 1: Asignación B1=Red, B2=Green, B3=Blue (color verdadero)
    rgb_opcion1 = np.stack([norm1, norm2, norm3], axis=2)
    axes[0].imshow(rgb_opcion1)
    axes[0].set_title('Opción 1: B1=Red, B2=Green, B3=Blue\n(Color Verdadero)')
    axes[0].axis('off')
    
    # Opción 2: Asignación B1=NIR, B2=Red, B3=Green (falso color)
    rgb_opcion2 = np.stack([norm1, norm2, norm3], axis=2)
    axes[1].imshow(rgb_opcion2)
    axes[1].set_title('Opción 2: B1=NIR, B2=Red, B3=Green\n(Falso Color Estándar)')
    axes[1].axis('off')
    
    # NDVI con la configuración que parece correcta
    im = axes[2].imshow(ndvi_test, cmap='RdYlGn', vmin=-1, vmax=1)
    axes[2].set_title(f'NDVI: {config}')
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "="*60)
    print("📌 INSTRUCCIÓN:")
    print("Observa las imágenes:")
    print("- En Color Verdadero: las áreas quemadas se ven oscuras/marrones")
    print("- En Falso Color Estándar: las áreas quemadas se ven rojizas/oscuras")
    print("- La vegetación sana en falso color se ve verde brillante")
    print("\n¿En qué imagen puedes identificar mejor las áreas quemadas?")
    print("Esa es la configuración correcta para tu análisis")
    print("="*60)