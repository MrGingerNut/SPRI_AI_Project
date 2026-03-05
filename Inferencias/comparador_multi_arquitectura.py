"""
Comparador Multi-Arquitectura de Detección de Incendios
========================================================
Compara UNet y Wildfire contra una imagen original (ground truth).

Formatos de imagen detectados automáticamente:
  - Original   : RGB .tif, incendios en ROJO (R>200, G<100, B<100)
  - Wildfire   : RGB .tif, incendios en rojo/blanco/amarillo (multi-color)
  - UNet       : 1 banda Byte, BINARIO 0=fondo / 1=incendio

Estructura de archivos esperada en la carpeta:
  Huamuchil_Oaxaca_UNet.tif        (o UNet2, UNet3, etc. con --sufijo)
  Huamuchil_Oaxaca_Wildfire.tif    (o Wildfire2, Wildfire3, etc.)

Uso:
  # Sin sufijo (UNet.tif, Wildfire.tif)
  python comparador_multi_arquitectura.py \
      --original  /ruta/Huamuchil_Oaxaca_Original.tif \
      --carpeta   /ruta/carpeta/ \
      --localidad Huamuchil_Oaxaca \
      --salida    /ruta/resultados/

  # Con sufijo numérico (UNet3.tif, Wildfire3.tif)
  python comparador_multi_arquitectura.py \
      --original  /ruta/Huamuchil_Oaxaca_Original.tif \
      --carpeta   /ruta/carpeta/ \
      --localidad Huamuchil_Oaxaca \
      --sufijo    3 \
      --salida    /ruta/resultados/

  # Sin abrir ventanas (solo guardar archivos)
  ... --no_mostrar
"""

import sys
import re
import argparse
from pathlib import Path
from datetime import datetime

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "0"
os.environ["QT_QPA_PLATFORM"] = "offscreen"   # evita que OpenCV intente abrir Qt

import cv2
import numpy as np
from osgeo import gdal
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec


# ===========================================================================
# CARGA
# ===========================================================================

def cargar_geotiff(ruta):
    ds = gdal.Open(str(ruta))
    if ds is None:
        raise ValueError(f"No se pudo abrir: {ruta}")
    bandas = [ds.GetRasterBand(i).ReadAsArray() for i in range(1, ds.RasterCount + 1)]
    img = np.stack(bandas, axis=-1) if len(bandas) > 1 else bandas[0]
    gt, prj = ds.GetGeoTransform(), ds.GetProjection()
    ds = None
    return img, gt, prj


# ===========================================================================
# DETECCIÓN DE INCENDIOS
# ===========================================================================

def detectar_original(imagen, umbral_rojo=200):
    """
    Ground truth RGB: incendios en ROJO puro.
    Detecta: R > umbral_rojo  AND  G < 100  AND  B < 100
    """
    if imagen.ndim == 2:
        # Fallback si llegara en grises
        mascara = imagen > umbral_rojo
    else:
        r, g, b = imagen[:, :, 0], imagen[:, :, 1], imagen[:, :, 2]
        mascara = (r > umbral_rojo) & (g < 100) & (b < 100)

    print(f"  [Original]  {mascara.sum():>10,} px de incendio  "
          f"({mascara.sum()/mascara.size*100:.2f}% del total)")
    return mascara, _mascara_a_rgb(mascara)


def detectar_wildfire(imagen, umbral_rojo=200):
    """
    Predicción Wildfire (RGB): puede tener rojos intensos o colores mixtos.
    Si hay rojos claros (>500 px) se usa solo esa máscara;
    si no, se combina rojo + blanco + amarillo + no-negro.
    """
    if imagen.ndim == 2:
        mascara = imagen > 128
        print(f"  [Wildfire]  Escala de grises (inesperado), umbral>128: {mascara.sum():,} px")
    else:
        r, g, b = imagen[:, :, 0], imagen[:, :, 1], imagen[:, :, 2]
        m_rojo     = (r > umbral_rojo) & (g < 50)  & (b < 50)
        m_blanco   = (r > 200)         & (g > 200)  & (b > 200)
        m_amarillo = (r > 180)         & (g > 100)  & (b < 100)
        m_no_negro = (r > 50)          | (g > 50)   | (b > 50)

        if m_rojo.sum() > 500:
            mascara = m_rojo
            modo = "ROJO"
        else:
            mascara = m_rojo | m_blanco | m_amarillo | m_no_negro
            modo = "MULTI-COLOR"

        print(f"  [Wildfire]  {mascara.sum():>10,} px de incendio  "
              f"(modo {modo})")

    return mascara, _mascara_a_rgb(mascara)


def detectar_unet(imagen):
    """
    Predicción UNet: banda única Byte, BINARIO 0=fondo / 1=incendio.
    Min=0, Max=1 confirmado por gdalinfo.
    """
    if imagen.ndim == 3:
        imagen = imagen[:, :, 0]   # tomar primera banda si viene apilado

    mascara = imagen.astype(bool)  # 0→False, 1→True

    print(f"  [UNet]      {mascara.sum():>10,} px de incendio  "
          f"({mascara.sum()/mascara.size*100:.2f}% del total)")
    return mascara, _mascara_a_rgb(mascara)


def _mascara_a_rgb(mascara):
    """Convierte máscara booleana a imagen RGB: fondo blanco, incendios rojos."""
    h, w = mascara.shape
    rgb = np.full((h, w, 3), 255, dtype=np.uint8)
    rgb[mascara] = [255, 0, 0]
    return rgb


def redimensionar_si_necesario(imagen, shape_ref, nombre):
    """Redimensiona la imagen al tamaño de referencia si difieren."""
    if imagen.shape[:2] != shape_ref[:2]:
        print(f"  [{nombre}] Redimensionando de {imagen.shape[:2]} a {shape_ref[:2]}")
        interp = cv2.INTER_NEAREST
        if imagen.ndim == 2:
            imagen = cv2.resize(imagen, (shape_ref[1], shape_ref[0]), interpolation=interp)
        else:
            imagen = cv2.resize(imagen, (shape_ref[1], shape_ref[0]), interpolation=interp)
    return imagen


# ===========================================================================
# MÉTRICAS
# ===========================================================================

def calcular_metricas(mascara_ref, mascara_pred):
    vp = int(( mascara_ref &  mascara_pred).sum())
    fp = int((~mascara_ref &  mascara_pred).sum())
    fn = int(( mascara_ref & ~mascara_pred).sum())
    vn = int((~mascara_ref & ~mascara_pred).sum())
    total = mascara_ref.size
    n_ref = int(mascara_ref.sum())

    precision     = vp / (vp + fp)              if (vp + fp) > 0 else 0.0
    recall        = vp / (vp + fn)              if (vp + fn) > 0 else 0.0
    f1            = 2*precision*recall / (precision+recall) if (precision+recall) > 0 else 0.0
    exactitud     = (vp + vn) / total
    especificidad = vn / (vn + fp)              if (vn + fp) > 0 else 0.0
    pixeles_error = fp + fn
    pct_error     = pixeles_error / n_ref * 100 if n_ref > 0 else pixeles_error / total * 100

    return dict(
        vp=vp, fp=fp, fn=fn, vn=vn, total=total,
        n_ref=n_ref, n_pred=int(mascara_pred.sum()),
        pixeles_error=int(pixeles_error), pct_error=pct_error,
        precision=precision, recall=recall, f1=f1,
        exactitud=exactitud, especificidad=especificidad,
        pct_vp=vp/total*100, pct_fp=fp/total*100,
        pct_fn=fn/total*100, pct_vn=vn/total*100,
    )


def imagen_diferencias(mascara_ref, mascara_pred):
    """
    Imagen RGB de diferencias sobre fondo blanco:
      Rojo        = Verdaderos Positivos  (VP)
      Azul oscuro = Falsos Negativos      (FN)
      Azul claro  = Falsos Positivos      (FP)
      Blanco      = Verdaderos Negativos  (VN)
    """
    h, w = mascara_ref.shape
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    img[ mascara_ref &  mascara_pred] = [255,   0,   0]   # VP
    img[ mascara_ref & ~mascara_pred] = [  0,   0, 150]   # FN
    img[~mascara_ref &  mascara_pred] = [  0,   0, 255]   # FP
    return img


# ===========================================================================
# FIGURA: 5 imágenes + gráfica de métricas
#
#  Fila 0 │ Original │ Wildfire (proc.) │ UNet (proc.)      │
#  Fila 1 │ Leyenda  │ Diff Wildfire    │ Diff UNet         │
#  Fila 2 │     Barras de métricas comparativas             │
# ===========================================================================

COLORES = {'SegNet': '#2980b9', 'UNet': '#c0392b'}


def visualizar(localidad, img_orig_rgb, d_wf, d_un, ruta_salida=None):
    fig = plt.figure(figsize=(18, 14))
    gs  = gridspec.GridSpec(3, 3, figure=fig,
                            height_ratios=[3, 3, 2.2],
                            hspace=0.40, wspace=0.08)

    # ── Fila 0: imágenes procesadas (fondo blanco, fuego rojo) ───────────────
    _panel(fig.add_subplot(gs[0, 0]), img_orig_rgb,
           'Pixeles etiquetados ESA',
           f'Pixeles de incendios : {d_wf["m"]["n_ref"]:,} px', 'black')

    _panel(fig.add_subplot(gs[0, 1]), d_wf['rgb'],
           'SegNet (predicción)',
           f'Incendios detectados: {d_wf["m"]["n_pred"]:,} px', COLORES['SegNet'])

    _panel(fig.add_subplot(gs[0, 2]), d_un['rgb'],
           'UNet  (predicción)',
           f'Incendios detectados: {d_un["m"]["n_pred"]:,} px', COLORES['UNet'])

    # ── Fila 1: leyenda + imágenes de diferencias ────────────────────────────
    ax_ley = fig.add_subplot(gs[1, 0])
    ax_ley.axis('off')
    patches = [
        mpatches.Patch(color='red',     label='Verdadero Positivo (VP)\nIncendio real, detectado'),
        mpatches.Patch(color='#000096', label='Falso Negativo (FN)\nIncendio real, NO detectado'),
        mpatches.Patch(color='blue',    label='Falso Positivo (FP)\nDetectado sin ser incendio'),
        mpatches.Patch(facecolor='white', edgecolor='gray',
                       label='Verdadero Negativo (VN)\nFondo correcto'),
    ]
    ax_ley.legend(handles=patches, loc='center', fontsize=10,
                  title='DIFERENCIAS', title_fontsize=11,
                  framealpha=0.95, edgecolor='gray')

    mwf = d_wf['m']
    _panel(fig.add_subplot(gs[1, 1]), d_wf['diff'],
           'Diferencias SegNet vs ESA',
           f'F1={mwf["f1"]*100:.1f}%   Recall={mwf["recall"]*100:.1f}%   '
           f'Precisión={mwf["precision"]*100:.1f}%', COLORES['SegNet'])

    mun = d_un['m']
    _panel(fig.add_subplot(gs[1, 2]), d_un['diff'],
           'Diferencias  UNet vs ESA',
           f'F1={mun["f1"]*100:.1f}%   Recall={mun["recall"]*100:.1f}%   '
           f'Precisión={mun["precision"]*100:.1f}%', COLORES['UNet'])

    # ── Fila 2: barras comparativas ──────────────────────────────────────────
    _barras(fig.add_subplot(gs[2, :]), mwf, mun)

    fig.suptitle(f'Comparación entre arquitecturas — {localidad}',
                 fontsize=15, fontweight='bold', y=1.01)

    if ruta_salida:
        fig.savefig(str(ruta_salida), dpi=200, bbox_inches='tight')
        print(f"  Figura guardada: {ruta_salida}")
    return fig


def _panel(ax, img_rgb, titulo, subtitulo, color_titulo):
    ax.imshow(img_rgb)
    ax.set_title(titulo, fontsize=11, fontweight='bold', color=color_titulo, pad=5)
    ax.set_xlabel(subtitulo, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def _barras(ax, mwf, mun):
    nombres = ['Precisión', 'Recall', 'F1-Score', 'Exactitud', 'Especificidad']
    keys    = ['precision', 'recall', 'f1', 'exactitud', 'especificidad']
    x, ancho = np.arange(len(nombres)), 0.32

    def dibujar(vals, offset, color, label):
        bars = ax.bar(x + offset, vals, ancho, label=label,
                      color=color, alpha=0.85, edgecolor='black', linewidth=0.6)
        for bar, v in zip(bars, vals):
            if v > 3:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                        f'{v:.1f}%', ha='center', va='bottom', fontsize=8.5)

    dibujar([mwf[k]*100 for k in keys], -ancho/2, COLORES['SegNet'], 'SegNet')
    dibujar([mun[k]*100 for k in keys], +ancho/2, COLORES['UNet'],     'UNet')

    ax.set_xticks(x); ax.set_xticklabels(nombres, fontsize=11)
    ax.set_ylim(0, 115)
    ax.set_ylabel('Porcentaje (%)', fontsize=11)
    ax.set_title('Comparación de resultados por arquitectura', fontsize=12, fontweight='bold')
    ax.axhline(80, color='Green',  linestyle='--', alpha=0.45, linewidth=1.2, label='Excelente ≥80%')
    ax.axhline(60, color='Orange', linestyle='--', alpha=0.45, linewidth=1.2, label='Aceptable ≥60%')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)


# # ===========================================================================
# # REPORTE DE TEXTO
# # ===========================================================================

# def guardar_reporte(ruta, localidad, mwf, mun):
#     sep = "=" * 70

#     def bloque(nombre, m, f):
#         f.write(f"\n{'─'*50}\nARQUITECTURA: {nombre}\n{'─'*50}\n")
#         f.write(f"Píxeles incendio (referencia) : {m['n_ref']:>10,}\n")
#         f.write(f"Píxeles incendio (predicción) : {m['n_pred']:>10,}\n")
#         f.write(f"Píxeles errados               : {m['pixeles_error']:>10,}  ({m['pct_error']:.2f}%)\n\n")
#         f.write(f"Matriz de confusión:\n")
#         f.write(f"  VP={m['vp']:,}  FP={m['fp']:,}  FN={m['fn']:,}  VN={m['vn']:,}\n\n")
#         f.write(f"Métricas:\n")
#         for lbl, k in [('Precisión','precision'), ('Recall','recall'), ('F1-Score','f1'),
#                         ('Exactitud','exactitud'), ('Especificidad','especificidad')]:
#             f.write(f"  {lbl:<15}: {m[k]*100:.2f}%\n")

#     with open(str(ruta), 'w', encoding='utf-8') as f:
#         f.write(f"{sep}\nREPORTE MULTI-ARQUITECTURA — {localidad}\n")
#         f.write(f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n{sep}\n")
#         bloque('Wildfire', mwf, f)
#         bloque('UNet',     mun, f)
#         f.write(f"\n{sep}\nFIN DEL REPORTE\n{sep}\n")

#     print(f"  Reporte guardado: {ruta}")


# ===========================================================================
# BÚSQUEDA DE ARCHIVOS
# ===========================================================================

def buscar_archivo(carpeta, localidad, arquitectura, sufijo):
    """Busca <localidad>_<Arquitectura><sufijo>.tif (insensible a mayúsculas)."""
    patron = re.compile(
        rf'^{re.escape(localidad)}[_\-]{re.escape(arquitectura)}{re.escape(sufijo)}\.tif$',
        re.IGNORECASE
    )
    for f in Path(carpeta).iterdir():
        if patron.match(f.name):
            return f
    return None


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Comparador UNet vs Wildfire vs Original'
    )
    parser.add_argument('--original',   required=True,
                        help='Ruta al GeoTIFF de ground truth (RGB, incendios en rojo)')
    parser.add_argument('--carpeta',    required=True,
                        help='Carpeta con los archivos de predicción')
    parser.add_argument('--localidad',  required=True,
                        help='Prefijo de localidad, ej: Huamuchil_Oaxaca')
    parser.add_argument('--sufijo',     default='',
                        help='Sufijo numérico, ej: 3  →  UNet3.tif, Wildfire3.tif')
    parser.add_argument('--salida',     default='resultados/',
                        help='Carpeta de salida (se crea si no existe)')
    parser.add_argument('--umbral_rojo', type=int, default=200,
                        help='Umbral canal rojo para Original y Wildfire (default 200)')
    parser.add_argument('--no_mostrar', action='store_true',
                        help='No abrir ventanas de matplotlib')
    args = parser.parse_args()

    salida = Path(args.salida)
    salida.mkdir(parents=True, exist_ok=True)
    sep = "=" * 60

    print(f"\n{sep}")
    print("COMPARADOR MULTI-ARQUITECTURA DE INCENDIOS")
    print(f"  Localidad : {args.localidad}")
    print(f"  Sufijo    : '{args.sufijo}' (vacío = sin número)")
    print(f"{sep}\n")

    # ── 1. Cargar y detectar imagen original ─────────────────────────────────
    print(f"Cargando original: {args.original}")
    img_orig, gt, prj = cargar_geotiff(args.original)
    mascara_orig, rgb_orig = detectar_original(img_orig, args.umbral_rojo)

    # ── 2. Buscar Wildfire ────────────────────────────────────────────────────
    ruta_wf = buscar_archivo(args.carpeta, args.localidad, 'Wildfire', args.sufijo)
    if ruta_wf is None:
        print(f"\n[ERROR] No se encontró Wildfire{args.sufijo}.tif en {args.carpeta}")
        sys.exit(1)
    print(f"\nCargando Wildfire: {ruta_wf.name}")
    img_wf, _, _ = cargar_geotiff(ruta_wf)
    img_wf = redimensionar_si_necesario(img_wf, img_orig.shape, 'Wildfire')
    mascara_wf, rgb_wf = detectar_wildfire(img_wf, args.umbral_rojo)

    # ── 3. Buscar UNet ────────────────────────────────────────────────────────
    ruta_un = buscar_archivo(args.carpeta, args.localidad, 'UNet', args.sufijo)
    if ruta_un is None:
        print(f"\n[ERROR] No se encontró UNet{args.sufijo}.tif en {args.carpeta}")
        sys.exit(1)
    print(f"\nCargando UNet    : {ruta_un.name}")
    img_un, _, _ = cargar_geotiff(ruta_un)
    img_un = redimensionar_si_necesario(img_un, img_orig.shape, 'UNet')
    mascara_un, rgb_un = detectar_unet(img_un)

    # ── 4. Calcular métricas y diferencias ────────────────────────────────────
    print(f"\nCalculando métricas...")
    m_wf   = calcular_metricas(mascara_orig, mascara_wf)
    diff_wf = imagen_diferencias(mascara_orig, mascara_wf)

    m_un   = calcular_metricas(mascara_orig, mascara_un)
    diff_un = imagen_diferencias(mascara_orig, mascara_un)

    d_wf = {'rgb': rgb_wf, 'diff': diff_wf, 'm': m_wf}
    d_un = {'rgb': rgb_un, 'diff': diff_un, 'm': m_un}

    # ── 5. Figura comparativa (5 imágenes + barras) ───────────────────────────
    nombre_fig = salida / f"{args.localidad}{args.sufijo}_comparacion.png"
    fig = visualizar(
        localidad=f"{args.localidad}",
        img_orig_rgb=rgb_orig,
        d_wf=d_wf, d_un=d_un,
        ruta_salida=nombre_fig
    )

    # # ── 6. Reporte de texto ───────────────────────────────────────────────────
    # guardar_reporte(
    #     salida / f"{args.localidad}{args.sufijo}_reporte.txt",
    #     args.localidad, m_wf, m_un
    # )

    # ── 7. Resumen en consola ─────────────────────────────────────────────────
    print(f"\n{sep}")
    print("RESUMEN DE MÉTRICAS")
    print(f"{'Métrica':<16} {'Wildfire':>10} {'UNet':>10}")
    print("─" * 38)
    for lbl, k in [('Precisión','precision'), ('Recall','recall'), ('F1-Score','f1'),
                    ('Exactitud','exactitud'), ('Especificidad','especificidad')]:
        print(f"  {lbl:<14} {m_wf[k]*100:>9.2f}% {m_un[k]*100:>9.2f}%")
    print(f"{sep}\n")
    print(f"Archivos generados en: {salida.resolve()}")

    if not args.no_mostrar:
        plt.show()


if __name__ == "__main__":
    main()