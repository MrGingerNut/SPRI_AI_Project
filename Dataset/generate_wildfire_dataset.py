import os
import zipfile
import subprocess
import glob
import re
import numpy as np
import shutil
import rasterio
from osgeo import gdal

# rutas de configuracion
input_directory = "/home/liese2/SPRI_AI_project/Dataset/Raw" 
output_directory = input_directory
temp_directory = input_directory

output_dir_img = "/home/liese2/SPRI_AI_project/Dataset/Crops/True"  # Imágenes TRUE de 4 bandas
output_dir_mask = "/home/liese2/SPRI_AI_project/Dataset/Crops/Mask"  # Máscaras de 3 bandas

# Configuracion para gdal_translate
target_min = 0
target_max = (2 ** 16) - 1
new_data_type = "UInt16"  

# formato final para las imagenes true
final_format = "GTiff"

# CONFIGURACIÓN DE PARÁMETROS
UMBRAL_PORCENTAJE_ROJO = 0.20  
TAMANO_BLOQUE = 128  
OVERLAP = 64  

# crear directorios
os.makedirs(output_directory, exist_ok=True)
os.makedirs(temp_directory, exist_ok=True)
os.makedirs(output_dir_img, exist_ok=True)
os.makedirs(output_dir_mask, exist_ok=True)

def find_file_in_zip(zip_ref, pattern):
    for file_name in zip_ref.namelist():
        if pattern in file_name and file_name.endswith(('.tif', '.tiff')):
            return file_name
    return None

def get_band_min_max(file_path):
    cmd = ["gdalinfo", "-mm", file_path]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        output = result.stdout
        
        match = re.search(r"Computed Min/Max=([0-9\.\-]+),([0-9\.\-]+)", output)
        
        if match:
            min_val = match.group(1)
            max_val = match.group(2)
            return min_val, max_val
        else:
            print(f"No se encontro Computed Min/Max en {os.path.basename(file_path)}")
            return None, None
            
    except Exception as e:
        print(f"Error ejecutando gdalinfo: {e}")
        return None, None

def translate_band(input_path, output_path, min_val, max_val):
    cmd = [
        "gdal_translate",
        input_path, output_path,
        "-scale", str(min_val), str(max_val), str(target_min), str(target_max),
        "-ot", new_data_type
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def crear_mascara_roja_pura(imagen_burned_area):
    """
    Crea una máscara con solo píxeles rojos puros en UInt16
    """
    try:
        with rasterio.open(imagen_burned_area) as src:
            if src.count >= 3:
                banda_r = src.read(1)
                banda_g = src.read(2)
                banda_b = src.read(3)
                
                # NORMALIZAR A UINT8 PARA DETECCIÓN
                def normalizar_banda(banda):
                    if banda.dtype != np.uint8:
                        if banda.dtype == np.uint16:
                            return (banda / 256).astype(np.uint8)
                        else:
                            banda_norm = (banda - banda.min()) / (banda.max() - banda.min()) * 255
                            return banda_norm.astype(np.uint8)
                    return banda
                
                banda_r_norm = normalizar_banda(banda_r)
                banda_g_norm = normalizar_banda(banda_g)
                banda_b_norm = normalizar_banda(banda_b)
                
                # DETECCIÓN ESTRICTA
                umbral_rojo = 254
                
                mascara_roja = (
                    (banda_r_norm >= umbral_rojo) &
                    (banda_r_norm > banda_g_norm + 40) &
                    (banda_r_norm > banda_b_norm + 40) &
                    (banda_r_norm > 50)
                )
                
                mascara_verde_bajo = (banda_g_norm < 100)
                mascara_azul_bajo = (banda_b_norm < 100)
                mascara_final = mascara_roja & mascara_verde_bajo & mascara_azul_bajo
                
                # Contar píxeles
                rojo_pixels = np.sum(mascara_final)
                total_pixels = mascara_final.size
                porcentaje_rojo = (rojo_pixels / total_pixels) * 100
                
                print(f"    Píxeles rojos detectados en imagen completa: {rojo_pixels} ({porcentaje_rojo:.2f}%)")
                
                if rojo_pixels == 0:
                    print("    ⚠  No se detectaron píxeles rojos puros")
                    return None
                
                # CREAR MÁSCARA EN UINT16
                mascara_r = np.where(mascara_final, np.uint16(65535), np.uint16(0))
                mascara_g = np.where(mascara_final, np.uint16(0), np.uint16(0))
                mascara_b = np.where(mascara_final, np.uint16(0), np.uint16(0))
                
                return {
                    'r': mascara_r,
                    'g': mascara_g,
                    'b': mascara_b,
                    'profile': src.profile,
                    'transform': src.transform,
                    'crs': src.crs
                }
            else:
                print(f"    Imagen tiene {src.count} bandas, se esperaban 3")
                return None
                
    except Exception as e:
        print(f"    Error creando máscara: {str(e)}")
        return None
  
    
def dividir_y_filtrar_imagen(ruta_imagen_true, ruta_imagen_mask, tamaño_bloque=TAMANO_BLOQUE, 
                            overlap=OVERLAP, zip_name=None):
    """
    Procesa la imagen de 4 bandas y la máscara para generar crops
    CON OVERLAP para capturar mejor las áreas rojas
    SOLO guarda bloques que tengan al menos el UMBRAL_PORCENTAJE_ROJO de píxeles rojos
    """
    try:
        print(f"\n    Procesando par de imágenes:")
        print(f"      TRUE (4 bandas): {os.path.basename(ruta_imagen_true)}")
        print(f"      MASK (3 bandas): {os.path.basename(ruta_imagen_mask)}")
        
        # 1. Crear máscara de píxeles rojos puros
        print(f"    Creando máscara de píxeles rojos puros...")
        mascara_data = crear_mascara_roja_pura(ruta_imagen_mask)
        
        if mascara_data is None:
            print("    ❌ No se pudo crear máscara válida")
            return 0
        
        # 2. Leer imagen verdadera de 4 bandas
        print(f"    Leyendo imagen verdadera de 4 bandas...")
        with rasterio.open(ruta_imagen_true) as src_true:
            if src_true.count < 4:
                print(f"    ❌ Imagen verdadera tiene {src_true.count} bandas, se esperaban 4")
                return 0
            
            # Leer las 4 bandas (R,G,B,NIR)
            bandas_true = []
            for i in range(1, 5):
                bandas_true.append(src_true.read(i))
            
            alto, ancho = bandas_true[0].shape
            print(f"    Tamaño imagen TRUE: {ancho} x {alto}")
            print(f"    Tamaño imagen MASK: {mascara_data['r'].shape[1]} x {mascara_data['r'].shape[0]}")
            print(f"    Tipo de datos TRUE: {bandas_true[0].dtype}")
            print(f"    Tipo de datos MASK: {mascara_data['r'].dtype}")
            
            # Verificar dimensiones
            if (alto != mascara_data['r'].shape[0]) or (ancho != mascara_data['r'].shape[1]):
                print(f"    ⚠  ADVERTENCIA: Dimensiones no coinciden")
                print(f"      TRUE: {alto}x{ancho}, MASK: {mascara_data['r'].shape[0]}x{mascara_data['r'].shape[1]}")
            
            # 3. Procesar bloques CON OVERLAP
            contador = 0
            descartadas = 0
            descartadas_por_umbral = 0
            
            # Área común
            alto_comun = min(alto, mascara_data['r'].shape[0])
            ancho_comun = min(ancho, mascara_data['r'].shape[1])
            
            # Calcular paso con overlap
            paso = tamaño_bloque - overlap
            
            total_pixels_por_bloque = tamaño_bloque * tamaño_bloque
            minimo_pixels_rojos = int(total_pixels_por_bloque * UMBRAL_PORCENTAJE_ROJO)
            
            print(f"    Configuración de bloques:")
            print(f"      Tamaño bloque: {tamaño_bloque}x{tamaño_bloque} = {total_pixels_por_bloque} píxeles")
            print(f"      Overlap: {overlap} píxeles ({overlap/tamaño_bloque*100:.0f}%)")
            print(f"      Paso: {paso} píxeles")
            print(f"      Mínimo píxeles rojos requeridos: {minimo_pixels_rojos} ({UMBRAL_PORCENTAJE_ROJO*100}%)")
            print(f"      Área común: {ancho_comun}x{alto_comun}")
            
            # Calcular límites para no salir de la imagen
            max_y = alto_comun - tamaño_bloque
            max_x = ancho_comun - tamaño_bloque
            
            if max_y < 0 or max_x < 0:
                print(f"    ❌ Imagen demasiado pequeña para bloques de {tamaño_bloque}x{tamaño_bloque}")
                return 0
            
            print(f"    Explorando bloques con overlap...")
            
            # Usar paso en lugar de tamaño_bloque para tener overlap
            y_coords = list(range(0, max_y + 1, paso))
            x_coords = list(range(0, max_x + 1, paso))
            
            # Si el último bloque no llega al borde, agregarlo
            if y_coords[-1] != max_y:
                y_coords.append(max_y)
            if x_coords[-1] != max_x:
                x_coords.append(max_x)
            
            print(f"    Posiciones Y: {len(y_coords)} bloques")
            print(f"    Posiciones X: {len(x_coords)} bloques")
            print(f"    Total potencial de bloques: {len(y_coords) * len(x_coords)}")
            
            for y_idx, y in enumerate(y_coords):
                for x_idx, x in enumerate(x_coords):
                    # Extraer bloque de la MÁSCARA
                    bloque_mascara_r = mascara_data['r'][y:y+tamaño_bloque, x:x+tamaño_bloque]
                    
                    # Contar píxeles rojos
                    pixeles_rojos = np.sum(bloque_mascara_r > 0)
                    porcentaje_rojo_bloque = pixeles_rojos / total_pixels_por_bloque
                    
                    # CONDICIÓN: Solo guardar si tiene al menos el porcentaje requerido
                    if pixeles_rojos > 0 and porcentaje_rojo_bloque >= UMBRAL_PORCENTAJE_ROJO:
                        # Extraer bloque de la imagen TRUE
                        bloque_true = []
                        for banda in bandas_true:
                            bloque = banda[y:y+tamaño_bloque, x:x+tamaño_bloque]
                            bloque_true.append(bloque)
                        
                        # Extraer bloques de máscara completa
                        bloque_mascara_g = mascara_data['g'][y:y+tamaño_bloque, x:x+tamaño_bloque]
                        bloque_mascara_b = mascara_data['b'][y:y+tamaño_bloque, x:x+tamaño_bloque]
                        
                        # Crear nombres de archivo
                        nombre_base = os.path.splitext(zip_name)[0] if zip_name else "bloque"
                        
                        # GUARDAR IMAGEN TRUE (4 bandas, UInt16)
                        nombre_true = f"{nombre_base}_bloque_{contador}_x{x}_y{y}.tiff"
                        ruta_true = os.path.join(output_dir_img, nombre_true)
                        
                        perfil_true = src_true.profile.copy()
                        perfil_true.update({
                            'height': tamaño_bloque,
                            'width': tamaño_bloque,
                            'count': 4,
                            'dtype': 'uint16',
                            'transform': rasterio.Affine(
                                perfil_true['transform'].a,
                                perfil_true['transform'].b,
                                perfil_true['transform'].c + x * perfil_true['transform'].a,
                                perfil_true['transform'].d,
                                perfil_true['transform'].e,
                                perfil_true['transform'].f + y * perfil_true['transform'].e
                            )
                        })
                        
                        with rasterio.open(ruta_true, 'w', **perfil_true) as dst:
                            for i, bloque_banda in enumerate(bloque_true):
                                dst.write(bloque_banda, i+1)
                        
                        # GUARDAR MÁSCARA (3 bandas, UInt16)
                        nombre_mask = f"{nombre_base}_bloque_{contador}_x{x}_y{y}.tiff"
                        ruta_mask = os.path.join(output_dir_mask, nombre_mask)
                        
                        perfil_mask = {
                            'driver': 'GTiff',
                            'height': tamaño_bloque,
                            'width': tamaño_bloque,
                            'count': 3,
                            'dtype': 'uint16',
                            'crs': mascara_data.get('crs'),
                            'transform': rasterio.Affine(
                                mascara_data['transform'].a,
                                mascara_data['transform'].b,
                                mascara_data['transform'].c + x * mascara_data['transform'].a,
                                mascara_data['transform'].d,
                                mascara_data['transform'].e,
                                mascara_data['transform'].f + y * mascara_data['transform'].e
                            )
                        }
                        
                        with rasterio.open(ruta_mask, 'w', **perfil_mask) as dst:
                            dst.write(bloque_mascara_r, 1)
                            dst.write(bloque_mascara_g, 2)
                            dst.write(bloque_mascara_b, 3)
                        
                        if contador < 10:  # Mostrar primeros 10 bloques
                            print(f"      ✓ Bloque {contador} en ({x},{y}): {pixeles_rojos} rojos ({porcentaje_rojo_bloque*100:.1f}%)")
                        elif contador == 10:
                            print(f"      ... mostrando solo primeros 10 bloques ...")
                        
                        contador += 1
                    else:
                        descartadas += 1
                        if pixeles_rojos > 0 and porcentaje_rojo_bloque > 0:
                            descartadas_por_umbral += 1
                            if contador < 5:  # Mostrar primeros 5 descartados
                                print(f"      ✗ Bloque en ({x},{y}) descartado: {pixeles_rojos} rojos ({porcentaje_rojo_bloque*100:.1f}%) < {UMBRAL_PORCENTAJE_ROJO*100}%")
            
            print(f"\n    Resumen detallado:")
            print(f"      Bloques explorados: {len(y_coords) * len(x_coords)}")
            print(f"      Bloques guardados: {contador} (con ≥{UMBRAL_PORCENTAJE_ROJO*100}% rojos)")
            print(f"      Bloques descartados: {descartadas}")
            print(f"        - Sin píxeles rojos: {descartadas - descartadas_por_umbral}")
            print(f"        - Con píxeles rojos pero <{UMBRAL_PORCENTAJE_ROJO*100}%: {descartadas_por_umbral}")
            
            if contador > 0:
                print(f"\n    Ejemplo de bloques guardados:")
                print(f"      Cada bloque es de {tamaño_bloque}x{tamaño_bloque} píxeles")
                print(f"      Equivalente a {tamaño_bloque}x{tamaño_bloque} píxeles por imagen")
                print(f"      {contador} imágenes TRUE de 4 bandas guardadas en: {output_dir_img}")
                print(f"      {contador} imágenes MASK de 3 bandas guardadas en: {output_dir_mask}")
            
            return contador
            
    except Exception as e:
        print(f"    Error procesando imágenes: {str(e)}")
        import traceback
        print(f"    Detalles: {traceback.format_exc()}")
        return 0
    
def procesar_zip(zip_path, estadisticas_globales):
    """
    Procesa un archivo ZIP completo y actualiza estadísticas
    """
    zip_name = os.path.basename(zip_path)
    base_name = os.path.splitext(zip_name)[0]
    
    print(f"\n" + "="*60)
    print(f"PROCESANDO: {zip_name}")
    print("="*60)
    
    extract_dir = os.path.join(temp_directory, base_name)
    imagen_combinada = os.path.join(output_directory, f"{base_name}_merged.tif")
    
    try:
        # Estadísticas para este ZIP
        stats_zip = {
            'nombre': zip_name,
            'archivos_encontrados': 0,
            'bandas_encontradas': 0,
            'imagen_combinada_creada': False,
            'bloques_generados': 0,
            'error': None,
            'valido': False
        }
        
        # 1. Extraer archivos del ZIP
        print(f"\n1. Extrayendo archivos...")
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        # 2. Buscar archivos necesarios
        print(f"\n2. Buscando archivos necesarios...")
        
        # Buscar imagen Burned_Area_Detection
        archivo_burned = None
        archivos_tiff = [f for f in os.listdir(extract_dir) if f.endswith(('.tif', '.tiff'))]
        stats_zip['archivos_encontrados'] = len(archivos_tiff)
        
        for archivo in archivos_tiff:
            if "Burned_Area_Detection" in archivo:
                archivo_burned = os.path.join(extract_dir, archivo)
                print(f"   ✓ Encontrada: {archivo}")
                break
        
        if not archivo_burned:
            print(f"   ❌ No se encontró archivo Burned_Area_Detection")
            stats_zip['error'] = "No se encontró archivo Burned_Area_Detection"
            estadisticas_globales['zips_invalidos'].append(stats_zip)
            return 0
        
        # Buscar bandas individuales
        bandas = {}
        patrones = {
            'B02': 'B02',
            'B03': 'B03', 
            'B04': 'B04',
            'B08': 'B08'
        }
        
        for key, patron in patrones.items():
            for archivo in archivos_tiff:
                if patron in archivo and 'Raw' in archivo:
                    bandas[key] = os.path.join(extract_dir, archivo)
                    print(f"   ✓ Encontrada banda {key}: {archivo}")
                    break
        
        # Verificar que tengamos todas las bandas
        stats_zip['bandas_encontradas'] = len(bandas)
        if len(bandas) != 4:
            print(f"   ❌ Faltan bandas. Encontradas: {list(bandas.keys())}")
            stats_zip['error'] = f"Faltan bandas. Encontradas: {list(bandas.keys())}"
            estadisticas_globales['zips_invalidos'].append(stats_zip)
            return 0
        
        # 3. Fusionar bandas en imagen de 4 bandas
        print(f"\n3. Fusionando bandas en imagen de 4 bandas...")
        
        # Crear archivos temporales escalados
        scaled_files = []
        orden_bandas = ['B04', 'B03', 'B02', 'B08']  # R, G, B, NIR
        
        for banda_key in orden_bandas:
            ruta_banda = bandas[banda_key]
            
            # Obtener min/max
            src_min, src_max = get_band_min_max(ruta_banda)
            if src_min is None:
                src_min, src_max = 0, 10000
            
            # Escalar banda
            temp_output = os.path.join(temp_directory, f"{base_name}_{banda_key}_scaled.tif")
            translate_band(ruta_banda, temp_output, src_min, src_max)
            scaled_files.append(temp_output)
        
        # Fusionar con gdal_merge
        merge_cmd = [
            "gdal_merge.py",
            "-separate",
            "-ot", "UInt16",
            "-of", "GTiff",
            "-o", imagen_combinada
        ]
        merge_cmd.extend(scaled_files)
        
        result = subprocess.run(merge_cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"   ✓ Imagen combinada creada: {os.path.basename(imagen_combinada)}")
            stats_zip['imagen_combinada_creada'] = True
            
            # Mostrar información de la imagen
            try:
                with rasterio.open(imagen_combinada) as src:
                    print(f"   ✓ Dimensiones: {src.width}x{src.height}")
                    print(f"   ✓ Bandas: {src.count}")
                    print(f"   ✓ Tipo de dato: {src.dtypes[0]}")
                    print(f"   ✓ Píxeles totales: {src.width * src.height:,}")
            except Exception as e:
                print(f"   ⚠  No se pudo leer información de la imagen: {e}")
        else:
            print(f"   ❌ Error fusionando bandas: {result.stderr}")
            stats_zip['error'] = f"Error en gdal_merge: {result.stderr}"
            estadisticas_globales['zips_invalidos'].append(stats_zip)
            return 0
        
        # Limpiar archivos temporales escalados
        for f in scaled_files:
            if os.path.exists(f):
                os.remove(f)
        
        # 4. Procesar imagen combinada con máscara
        print(f"\n4. Generando crops de {TAMANO_BLOQUE}x{TAMANO_BLOQUE}...")
        bloques_generados = dividir_y_filtrar_imagen(
            ruta_imagen_true=imagen_combinada,
            ruta_imagen_mask=archivo_burned,
            tamaño_bloque=TAMANO_BLOQUE,
            overlap=OVERLAP,
            zip_name=zip_name
        )
        
        stats_zip['bloques_generados'] = bloques_generados
        
        # 5. Limpiar archivos temporales
        print(f"\n5. Limpiando archivos temporales...")
        if os.path.exists(extract_dir):
            shutil.rmtree(extract_dir)
        if os.path.exists(imagen_combinada):
            os.remove(imagen_combinada)
        
        print(f"\n" + "="*60)
        print(f"COMPLETADO: {zip_name}")
        print(f"Bloques generados: {bloques_generados}")
        if bloques_generados > 0:
            print(f"Cada bloque: {TAMANO_BLOQUE}x{TAMANO_BLOQUE} = {TAMANO_BLOQUE*TAMANO_BLOQUE} píxeles")
            print(f"Mínimo rojos por bloque: {int(TAMANO_BLOQUE*TAMANO_BLOQUE*UMBRAL_PORCENTAJE_ROJO)} píxeles")
            stats_zip['valido'] = True
            estadisticas_globales['zips_validos'].append(stats_zip)
        else:
            print(f"⚠  ZIP procesado pero no generó bloques válidos")
            stats_zip['valido'] = False
            stats_zip['error'] = "No generó bloques válidos"
            estadisticas_globales['zips_invalidos'].append(stats_zip)
        
        print("="*60)
        
        return bloques_generados
        
    except Exception as e:
        print(f"Error procesando {zip_name}: {str(e)}")
        stats_zip['error'] = str(e)
        estadisticas_globales['zips_invalidos'].append(stats_zip)
        return 0

def mostrar_estadisticas_detalladas(estadisticas_globales, total_bloques):
    """
    Muestra estadísticas detalladas del procesamiento
    """
    print(f"\n" + "="*80)
    print(f"ESTADÍSTICAS DETALLADAS DEL PROCESAMIENTO")
    print("="*80)
    
    total_zips = estadisticas_globales['total_zips']
    zips_validos = len(estadisticas_globales['zips_validos'])
    zips_invalidos = len(estadisticas_globales['zips_invalidos'])
    
    print(f"\nRESUMEN GENERAL:")
    print(f"  Total de archivos ZIP encontrados: {total_zips}")
    print(f"  Archivos ZIP válidos: {zips_validos} ({zips_validos/total_zips*100:.1f}%)")
    print(f"  Archivos ZIP inválidos: {zips_invalidos} ({zips_invalidos/total_zips*100:.1f}%)")
    print(f"  Total de bloques generados: {total_bloques}")
    
    if zips_validos > 0:
        print(f"\nZIPS VÁLIDOS ({zips_validos} archivos):")
        print("-" * 80)
        print(f"{'No.':<4} {'Nombre ZIP':<40} {'Bloques':<10} {'Archivos':<10} {'Bandas':<10}")
        print("-" * 80)
        
        total_bloques_validos = 0
        for i, zip_info in enumerate(estadisticas_globales['zips_validos'], 1):
            print(f"{i:<4} {zip_info['nombre'][:40]:<40} {zip_info['bloques_generados']:<10} "
                  f"{zip_info['archivos_encontrados']:<10} {zip_info['bandas_encontradas']:<10}")
            total_bloques_validos += zip_info['bloques_generados']
        
        print("-" * 80)
        print(f"TOTAL: {zips_validos} ZIPs válidos con {total_bloques_validos} bloques")
        
        # Estadísticas adicionales
        if zips_validos > 0:
            promedio_bloques = total_bloques_validos / zips_validos
            print(f"\nESTADÍSTICAS DE ZIPS VÁLIDOS:")
            print(f"  Promedio de bloques por ZIP: {promedio_bloques:.1f}")
            print(f"  Total de imágenes TRUE generadas: {total_bloques_validos}")
            print(f"  Total de imágenes MASK generadas: {total_bloques_validos}")
            print(f"  Total de imágenes (TRUE + MASK): {total_bloques_validos * 2}")
    
    if zips_invalidos > 0:
        print(f"\nZIPS INVÁLIDOS ({zips_invalidos} archivos):")
        print("-" * 80)
        print(f"{'No.':<4} {'Nombre ZIP':<40} {'Error':<40}")
        print("-" * 80)
        
        for i, zip_info in enumerate(estadisticas_globales['zips_invalidos'], 1):
            error_msg = zip_info['error'][:40] if zip_info['error'] else "Desconocido"
            print(f"{i:<4} {zip_info['nombre'][:40]:<40} {error_msg:<40}")
        
        print("-" * 80)
        print(f"TOTAL: {zips_invalidos} ZIPs inválidos")
    
    print(f"\nCONFIGURACIÓN UTILIZADA:")
    print(f"  Tamaño de bloque: {TAMANO_BLOQUE}x{TAMANO_BLOQUE} píxeles")
    print(f"  Overlap: {OVERLAP} píxeles ({OVERLAP/TAMANO_BLOQUE*100:.0f}%)")
    print(f"  Mínimo píxeles rojos por bloque: {UMBRAL_PORCENTAJE_ROJO*100}%")
    print(f"  Píxeles por bloque: {TAMANO_BLOQUE*TAMANO_BLOQUE}")
    print(f"  Mínimo rojos requeridos por bloque: {int(TAMANO_BLOQUE*TAMANO_BLOQUE*UMBRAL_PORCENTAJE_ROJO)}")
    
    print(f"\nDIRECTORIOS DE SALIDA:")
    print(f"  Imágenes TRUE (4 bandas UInt16): {output_dir_img}")
    print(f"  Imágenes MASK (3 bandas UInt16): {output_dir_mask}")
    print("="*80)

def main():
    zip_files = glob.glob(os.path.join(input_directory, "*.zip"))
    
    if not zip_files:
        print("No se encontraron archivos .zip")
        return
    
    # Inicializar estadísticas globales
    estadisticas_globales = {
        'total_zips': len(zip_files),
        'zips_validos': [],  # ZIPs que generaron bloques
        'zips_invalidos': [],  # ZIPs con errores o sin bloques
        'total_bloques': 0
    }
    
    print(f"\n" + "="*60)
    print(f"INICIANDO PROCESAMIENTO DE DATASET")
    print("="*60)
    print(f"Archivos ZIP encontrados: {len(zip_files)}")
    print(f"\nCONFIGURACIÓN:")
    print(f"  Tamaño de bloque: {TAMANO_BLOQUE}x{TAMANO_BLOQUE} píxeles")
    print(f"  Overlap: {OVERLAP} píxeles ({OVERLAP/TAMANO_BLOQUE*100:.0f}%)")
    print(f"  Mínimo píxeles rojos por bloque: {UMBRAL_PORCENTAJE_ROJO*100}%")
    print(f"  Píxeles por bloque: {TAMANO_BLOQUE*TAMANO_BLOQUE}")
    print(f"  Mínimo rojos requeridos: {int(TAMANO_BLOQUE*TAMANO_BLOQUE*UMBRAL_PORCENTAJE_ROJO)} píxeles")
    print(f"\nDirectorios de salida:")
    print(f"  TRUE (4 bandas UInt16): {output_dir_img}")
    print(f"  MASK (3 bandas UInt16): {output_dir_mask}")
    print("="*60)
    
    total_bloques = 0
    
    print(f"\nPROCESANDO {len(zip_files)} ARCHIVOS ZIP...")
    
    for i, zip_path in enumerate(zip_files, 1):
        print(f"\n[{i}/{len(zip_files)}] ", end="")
        bloques = procesar_zip(zip_path, estadisticas_globales)
        total_bloques += bloques
    
    # Mostrar estadísticas detalladas
    mostrar_estadisticas_detalladas(estadisticas_globales, total_bloques)
    
    # Resumen final
    print(f"\n" + "="*60)
    print(f"PROCESO GLOBAL COMPLETADO")
    print(f"Total de archivos ZIP procesados: {estadisticas_globales['total_zips']}")
    print(f"Archivos ZIP válidos: {len(estadisticas_globales['zips_validos'])}")
    print(f"Archivos ZIP inválidos: {len(estadisticas_globales['zips_invalidos'])}")
    print(f"Total de bloques generados: {total_bloques}")
    print(f"Total de imágenes TRUE (4 bandas): {total_bloques}")
    print(f"Total de imágenes MASK (3 bandas): {total_bloques}")
    print(f"Tamaño de cada imagen: {TAMANO_BLOQUE}x{TAMANO_BLOQUE} píxeles")
    print("="*60)

if __name__ == "__main__":
    main()