import rasterio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import glob

def verify_5bands(image_path):
    """Verifica estructura, valores y calidad del NDVI en una imagen de 5 bandas."""
    
    print(f"\n{'='*60}")
    print(f"Archivo: {os.path.basename(image_path)}")
    print(f"{'='*60}")
    
    with rasterio.open(image_path) as src:
        
        # --- 1. ESTRUCTURA ---
        print(f"\n📁 ESTRUCTURA:")
        print(f"  Número de bandas : {src.count}")
        print(f"  Dimensiones      : {src.width} x {src.height} px")
        print(f"  Tipo de dato     : {src.dtypes[0]}")
        print(f"  CRS              : {src.crs}")
        print(f"  Transform        : {src.transform}")
        
        if src.count != 5:
            print(f"  ❌ ERROR: Se esperaban 5 bandas, hay {src.count}")
            return
        else:
            print(f"  ✅ Bandas correctas: 5")
        
        # Leer todas las bandas
        bands = src.read()  # (5, H, W)
        band_names = ['Red (B04)', 'Green (B03)', 'Blue (B02)', 'NIR (B08)', 'NDVI']
        
        # --- 2. ESTADÍSTICAS POR BANDA ---
        print(f"\n📊 ESTADÍSTICAS POR BANDA:")
        print(f"  {'Banda':<15} {'Min':>8} {'Max':>8} {'Media':>10} {'Std':>10} {'NaN':>6}")
        print(f"  {'-'*60}")
        
        for i, name in enumerate(band_names):
            band = bands[i].astype(np.float32)
            nan_count = np.sum(np.isnan(band))
            print(f"  {name:<15} {band.min():>8.1f} {band.max():>8.1f} "
                  f"{band.mean():>10.2f} {band.std():>10.2f} {nan_count:>6}")
        
        # --- 3. VALIDACIÓN ESPECÍFICA DEL NDVI ---
        print(f"\n🌿 VALIDACIÓN NDVI (banda 5):")
        
        ndvi_raw = bands[4].astype(np.float32)
        
        # Revertir escala UInt16 → float [-1, 1]
        ndvi_float = (ndvi_raw / 65535.0) * 2.0 - 1.0
        
        print(f"  Rango UInt16     : [{ndvi_raw.min():.0f}, {ndvi_raw.max():.0f}]")
        print(f"  Rango float real : [{ndvi_float.min():.4f}, {ndvi_float.max():.4f}]")
        
        # Verificar rango válido
        if ndvi_float.min() >= -1.0 and ndvi_float.max() <= 1.0:
            print(f"  ✅ Rango NDVI válido [-1, 1]")
        else:
            print(f"  ❌ Rango NDVI fuera de [-1, 1]")
        
        # Distribución por categorías
        total_px = ndvi_float.size
        print(f"\n  Distribución por categoría:")
        categorias = [
            ("Agua / Sin datos  (NDVI < -0.1) ", ndvi_float < -0.1),
            ("Suelo desnudo     (-0.1 a 0.2)  ", (ndvi_float >= -0.1) & (ndvi_float < 0.2)),
            ("Vegetación escasa (0.2 a 0.5)   ", (ndvi_float >= 0.2)  & (ndvi_float < 0.5)),
            ("Vegetación densa  (NDVI > 0.5)  ", ndvi_float >= 0.5),
        ]
        for label, mask in categorias:
            pct = mask.sum() / total_px * 100
            print(f"    {label}: {pct:5.1f}%")
        
        # Coherencia NIR vs RED → NDVI
        print(f"\n  Coherencia NIR/RED/NDVI:")
        nir = bands[3].astype(np.float32)
        red = bands[0].astype(np.float32)
        ndvi_recalc = (nir - red) / (nir + red + 1e-8)
        ndvi_recalc = np.clip(ndvi_recalc, -1.0, 1.0)
        ndvi_recalc_uint16 = ((ndvi_recalc + 1.0) / 2.0 * 65535).astype(np.float32)
        
        diff = np.abs(ndvi_raw - ndvi_recalc_uint16)
        print(f"    Diferencia media vs recalculado : {diff.mean():.4f}")
        print(f"    Diferencia máxima               : {diff.max():.4f}")
        
        if diff.mean() < 1.0:
            print(f"    ✅ NDVI coherente con NIR y RED")
        else:
            print(f"    ⚠️  NDVI tiene discrepancia con NIR/RED")
        
        return bands, ndvi_float


def plot_5bands(image_path):
    """Visualiza las 5 bandas y un mapa de color del NDVI."""
    
    with rasterio.open(image_path) as src:
        bands = src.read()
    
    band_names = ['Red (B04)', 'Green (B03)', 'Blue (B02)', 'NIR (B08)', 'NDVI (raw)']
    
    fig = plt.figure(figsize=(18, 10))
    gs  = gridspec.GridSpec(2, 4, figure=fig)
    
    # Bandas 1-4 en escala de grises
    axes_bands = [fig.add_subplot(gs[0, i]) for i in range(4)]
    for i, (ax, name) in enumerate(zip(axes_bands, band_names[:4])):
        ax.imshow(bands[i], cmap='gray', vmin=0, vmax=65535)
        ax.set_title(name, fontsize=11)
        ax.axis('off')
    
    # NDVI en escala de color (RdYlGn)
    ax_ndvi_raw = fig.add_subplot(gs[0, 3])  # ya ocupado, usamos fila 1
    ax_ndvi_raw = fig.add_subplot(gs[1, 0])
    ndvi_float  = (bands[4].astype(np.float32) / 65535.0) * 2.0 - 1.0
    im = ax_ndvi_raw.imshow(ndvi_float, cmap='RdYlGn', vmin=-1, vmax=1)
    ax_ndvi_raw.set_title('NDVI [-1, 1]', fontsize=11)
    ax_ndvi_raw.axis('off')
    plt.colorbar(im, ax=ax_ndvi_raw, fraction=0.046)
    
    # Histograma NDVI
    ax_hist = fig.add_subplot(gs[1, 1:3])
    ax_hist.hist(ndvi_float.ravel(), bins=100, color='forestgreen', alpha=0.7, edgecolor='black', linewidth=0.3)
    ax_hist.axvline(-0.1, color='blue',   linestyle='--', label='Agua < -0.1')
    ax_hist.axvline(0.2,  color='orange', linestyle='--', label='Veg. escasa > 0.2')
    ax_hist.axvline(0.5,  color='red',    linestyle='--', label='Veg. densa > 0.5')
    ax_hist.set_xlabel('Valor NDVI')
    ax_hist.set_ylabel('Frecuencia')
    ax_hist.set_title('Histograma NDVI')
    ax_hist.legend()
    
    # RGB compuesto
    ax_rgb = fig.add_subplot(gs[1, 3])
    rgb = np.stack([bands[0], bands[1], bands[2]], axis=-1).astype(np.float32)
    rgb = (rgb / 65535.0 * 255).clip(0, 255).astype(np.uint8)
    ax_rgb.imshow(rgb)
    ax_rgb.set_title('RGB compuesto')
    ax_rgb.axis('off')
    
    plt.suptitle(f"Verificación 5 bandas: {os.path.basename(image_path)}", fontsize=13, fontweight='bold')
    plt.tight_layout()
    
    out_png = image_path.replace('.tif', '_verificacion.png')
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    print(f"\n  💾 Imagen guardada: {out_png}")
    plt.show()


def verify_all(output_directory):
    """Verifica todas las imágenes del directorio."""
    tifs = glob.glob(os.path.join(output_directory, "*.tif"))
    
    if not tifs:
        print("No se encontraron archivos .tif")
        return
    
    print(f"Encontrados {len(tifs)} archivos .tif")
    
    for tif_path in tifs:
        result = verify_5bands(tif_path)
        if result:
            plot_5bands(tif_path)


# --- EJECUTAR ---
output_directory = "/home/felix/SPRI_AI_Project/Dataset/Merged_5bands"
verify_all(output_directory)