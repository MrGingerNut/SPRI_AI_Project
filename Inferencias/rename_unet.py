#!/usr/bin/env python3
"""
Renombrador de archivos UNet
============================
Reglas de cambio de nombre:
  UNet   -> UNet5
  UNet2  -> UNet
  UNet3  -> sin cambio
  UNet4  -> UNet2
  UNet5  -> UNet4

Uso:
  python rename_unet.py /ruta/carpeta          # modo prueba (no cambia nada)
  python rename_unet.py /ruta/carpeta --apply  # aplica los cambios reales
"""

import os
import re
import sys
import argparse

# Orden: del numero mas alto al mas bajo para evitar sustituciones en cadena
# Cada tupla: (patron, reemplazo)  --  None = no cambiar
RULES = [
    (r'(?i)(unet)(5)(?=\D|$)', r'\g<1>4'),   # UNet5 -> UNet4
    (r'(?i)(unet)(4)(?=\D|$)', r'\g<1>2'),   # UNet4 -> UNet2
    (r'(?i)(unet)(3)(?=\D|$)', None),         # UNet3 -> sin cambio
    (r'(?i)(unet)(2)(?=\D|$)', r'\g<1>'),     # UNet2 -> UNet (sin numero)
    (r'(?i)(unet)(?!\d)',       r'\g<1>5'),   # UNet  -> UNet5
]


def compute_new_name(filename):
    """Devuelve el nuevo nombre si hay cambio, None si no aplica ninguna regla."""
    name, ext = os.path.splitext(filename)

    for pattern, replacement in RULES:
        if replacement is None:
            if re.search(pattern, name):
                return None   # UNet3: no tocar
            continue

        new_name, count = re.subn(pattern, replacement, name, count=1)
        if count:
            return new_name + ext

    return None


def process_folder(folder, apply):
    if not os.path.isdir(folder):
        print("[ERROR] La carpeta '{}' no existe.".format(folder))
        sys.exit(1)

    files = sorted(os.listdir(folder))
    changes = []

    for fname in files:
        result = compute_new_name(fname)
        if result is not None and result != fname:
            changes.append((fname, result))

    if not changes:
        print("No se encontraron archivos que requieran renombrar.")
        return

    mode_label = "REAL" if apply else "PRUEBA (usa --apply para aplicar cambios reales)"
    print("\n" + "="*60)
    print("  Modo     : {}".format(mode_label))
    print("  Carpeta  : {}".format(folder))
    print("  Archivos : {}".format(len(changes)))
    print("="*60 + "\n")

    for old, new in changes:
        print("  ANTES  : {}".format(old))
        print("  DESPUES: {}".format(new))
        if apply:
            src = os.path.join(folder, old)
            dst = os.path.join(folder, new)
            if os.path.exists(dst):
                print("  [OMITIDO] Ya existe el archivo destino.")
            else:
                os.rename(src, dst)
                print("  [OK] Renombrado")
        print()

    if not apply:
        print(">> Modo PRUEBA finalizado. Ningun archivo fue modificado.")
        print(">> Agrega --apply para realizar los cambios reales.\n")
    else:
        print(">> {} archivo(s) renombrado(s) exitosamente.\n".format(len(changes)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Renombra archivos cambiando el sufijo numerico de UNet."
    )
    parser.add_argument(
        "carpeta",
        help="Ruta de la carpeta con los archivos a renombrar"
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Aplica los cambios reales (por defecto solo muestra una vista previa)"
    )
    args = parser.parse_args()
    process_folder(args.carpeta, args.apply)
