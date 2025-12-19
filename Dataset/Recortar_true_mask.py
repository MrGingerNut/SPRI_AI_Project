import os
import numpy as np
import shutil
from PIL import Image
import rasterio

def dividir_y_filtrar_imagen(ruta_imagen, ruta_imagen_paralela=None, tamaño_bloque=128, carpeta_salida="bloques", umbral_rojo=100):
    try:
        print(f"    Leyendo imagen principal: {os.path.basename(ruta_imagen)}")
        
        # Leer la imagen principal con rasterio
        with rasterio.open(ruta_imagen) as src:
            # Leer todos los canales disponibles
            num_bandas = src.count
            print(f"     Bandas disponibles: {num_bandas}")
            
            # Guardar el perfil completo de la imagen principal
            perfil_principal = src.profile
            
            if num_bandas >= 4:
                banda_roja = src.read(3)
                banda_verde = src.read(2)
                banda_azul = src.read(1)
                banda_infrarroja = src.read(4)
                print(f"     Usando bandas: Rojo(3), Verde(2), Azul(1), Infrarrojo(4)")
            elif num_bandas == 3:
                banda_roja = src.read(1)
                banda_verde = src.read(2)
                banda_azul = src.read(3)
                banda_infrarroja = None
                print(f"     Usando bandas RGB estándar")
            else:
                print(f"     No hay suficientes bandas: {num_bandas}")
                return 0

        # Leer imagen paralela si se proporciona
        imagen_paralela_data = None
        if ruta_imagen_paralela and os.path.exists(ruta_imagen_paralela):
            print(f"    Leyendo imagen paralela: {os.path.basename(ruta_imagen_paralela)}")
            with rasterio.open(ruta_imagen_paralela) as src_paralela:
                num_bandas_paralela = src_paralela.count
                print(f"     Bandas disponibles en imagen paralela: {num_bandas_paralela}")
                
                # Guardar el perfil completo para conservar metadatos
                perfil_paralela = src_paralela.profile
                
                bandas_paralelas = []
                for i in range(1, num_bandas_paralela + 1):
                    bandas_paralelas.append(src_paralela.read(i))
                
                imagen_paralela_data = {
                    'bandas': bandas_paralelas,
                    'count': num_bandas_paralela,
                    'shape': src_paralela.shape,
                    'profile': perfil_paralela,  # Conservar metadatos
                    'dtype': src_paralela.dtypes[0]  # Conservar tipo de dato original
                }

        # Convertir a uint8 si es necesario para procesamiento
        def normalizar_banda(banda):
            if banda.dtype != np.uint8:
                if banda.dtype == np.uint16:
                    return (banda / 256).astype(np.uint8)
                else:
                    banda_norm = (banda - banda.min()) / (banda.max() - banda.min()) * 255
                    return banda_norm.astype(np.uint8)
            return banda

        banda_roja = normalizar_banda(banda_roja)
        banda_verde = normalizar_banda(banda_verde)
        banda_azul = normalizar_banda(banda_azul)
        
        if banda_infrarroja is not None:
            banda_infrarroja = normalizar_banda(banda_infrarroja)
            
        # Normalizar bandas de imagen paralela si existe (solo para procesamiento)
        if imagen_paralela_data:
            bandas_paralelas_normalizadas = []
            for banda in imagen_paralela_data['bandas']:
                bandas_paralelas_normalizadas.append(normalizar_banda(banda))
            imagen_paralela_data['bandas_normalizadas'] = bandas_paralelas_normalizadas

        # Crear imagen RGB para visualización y detección
        alto, ancho = banda_roja.shape
        print(f"     Tamaño de imagen principal: {ancho} x {alto}")
        if imagen_paralela_data:
            print(f"     Tamaño de imagen paralela: {imagen_paralela_data['shape'][1]} x {imagen_paralela_data['shape'][0]}")

        # Crear carpetas de salida
        carpeta_salida_principal = os.path.join(carpeta_salida, "principal")
        carpeta_salida_paralela = os.path.join(carpeta_salida, "paralela")
        os.makedirs(carpeta_salida_principal, exist_ok=True)
        if imagen_paralela_data:
            os.makedirs(carpeta_salida_paralela, exist_ok=True)

        contador = 0
        descartadas = 0
        paralelas_guardadas = 0

        for y in range(0, alto, tamaño_bloque):
            for x in range(0, ancho, tamaño_bloque):
                # Coordenadas del recorte
                y_end = min(y + tamaño_bloque, alto)
                x_end = min(x + tamaño_bloque, ancho)
                
                # Extraer bloques de cada banda de la imagen principal
                bloque_rojo = banda_roja[y:y_end, x:x_end]
                bloque_verde = banda_verde[y:y_end, x:x_end]
                bloque_azul = banda_azul[y:y_end, x:x_end]
                
                if banda_infrarroja is not None:
                    bloque_infrarrojo = banda_infrarroja[y:y_end, x:x_end]

                # Descartar si no tiene tamaño exacto
                if bloque_rojo.shape != (tamaño_bloque, tamaño_bloque):
                    descartadas += 1
                    continue

                # Crear imagen RGB combinada
                imagen_rgb = np.stack([bloque_rojo, bloque_verde, bloque_azul], axis=2)
                
                # Detectar presencia de color rojo
                mascara_roja = (
                    (bloque_rojo > umbral_rojo) & 
                    (bloque_rojo > bloque_verde + 40) & 
                    (bloque_rojo > bloque_azul + 40) &
                    (bloque_rojo > 50)
                )

                # Si hay al menos un píxel rojo, se guardan ambos bloques
                if np.any(mascara_roja):
                    # Crear imagen principal con ROJO UNIFORME
                    imagen_filtrada = np.zeros_like(imagen_rgb)
                    imagen_filtrada[mascara_roja] = [255, 0, 0]
                    
                    # GUARDAR IMAGEN PRINCIPAL con referencia espacial
                    nombre_bloque_principal = f"bloque_{contador}_x{x}_y{y}.tiff"
                    ruta_salida_principal = os.path.join(carpeta_salida_principal, nombre_bloque_principal)
                    
                    # Actualizar perfil para el bloque principal
                    perfil_principal_actualizado = perfil_principal.copy()
                    perfil_principal_actualizado.update({
                        'height': tamaño_bloque,
                        'width': tamaño_bloque,
                        'count': 3,  # RGB
                        'dtype': 'uint8'
                    })
                    
                    # Ajustar la transformación para las coordenadas del bloque
                    if perfil_principal_actualizado.get('transform'):
                        transform_original = perfil_principal_actualizado['transform']
                        # Calcular nueva transformación para este bloque específico
                        nueva_transform = rasterio.Affine(
                            transform_original.a,
                            transform_original.b,
                            transform_original.c + x * transform_original.a,
                            transform_original.d,
                            transform_original.e,
                            transform_original.f + y * transform_original.e
                        )
                        perfil_principal_actualizado['transform'] = nueva_transform
                    
                    # Guardar imagen principal con rasterio para conservar referencia espacial
                    with rasterio.open(ruta_salida_principal, 'w', **perfil_principal_actualizado) as dst:
                        # Reorganizar bandas para RGB
                        dst.write(imagen_filtrada[:, :, 0], 1)  # Rojo
                        dst.write(imagen_filtrada[:, :, 1], 2)  # Verde  
                        dst.write(imagen_filtrada[:, :, 2], 3)  # Azul
                    
                    # GUARDAR IMAGEN PARALELA - SIEMPRE que se detecte rojo en la principal
                    if imagen_paralela_data:
                        # Extraer bloques de la imagen paralela en las MISMAS coordenadas
                        bloques_paralelos = []
                        for i, banda in enumerate(imagen_paralela_data['bandas']):
                            alto_paralela, ancho_paralela = banda.shape
                            
                            # Asegurar que las coordenadas estén dentro de los límites
                            if y < alto_paralela and x < ancho_paralela:
                                y_end_paralela = min(y + tamaño_bloque, alto_paralela)
                                x_end_paralela = min(x + tamaño_bloque, ancho_paralela)
                                
                                # Usar datos ORIGINALES (no normalizados) para guardar
                                bloque_paralelo = banda[y:y_end_paralela, x:x_end_paralela]
                                
                                # Si el bloque es más pequeño, rellenar con ceros
                                if bloque_paralelo.shape != (tamaño_bloque, tamaño_bloque):
                                    bloque_completo = np.zeros((tamaño_bloque, tamaño_bloque), dtype=bloque_paralelo.dtype)
                                    bloque_completo[:bloque_paralelo.shape[0], :bloque_paralelo.shape[1]] = bloque_paralelo
                                    bloques_paralelos.append(bloque_completo)
                                else:
                                    bloques_paralelos.append(bloque_paralelo)
                            else:
                                # Si las coordenadas están fuera de límites, crear bloque negro
                                bloque_completo = np.zeros((tamaño_bloque, tamaño_bloque), dtype=imagen_paralela_data['dtype'])
                                bloques_paralelos.append(bloque_completo)
                        
                        # Crear y guardar imagen paralela con TODAS las bandas originales
                        if len(bloques_paralelos) > 0:
                            # Apilar las bandas en el formato (count, height, width)
                            datos_salida = np.stack(bloques_paralelos, axis=0)
                            
                            # Actualizar el perfil para el bloque recortado
                            perfil_actualizado = imagen_paralela_data['profile'].copy()
                            perfil_actualizado.update({
                                'height': tamaño_bloque,
                                'width': tamaño_bloque,
                                'count': len(bloques_paralelos),
                                'dtype': datos_salida.dtype
                            })
                            
                            # Ajustar la transformación para las coordenadas del bloque (igual que la principal)
                            if perfil_actualizado.get('transform'):
                                transform_original = perfil_actualizado['transform']
                                # Calcular nueva transformación para este bloque específico
                                nueva_transform = rasterio.Affine(
                                    transform_original.a,
                                    transform_original.b,
                                    transform_original.c + x * transform_original.a,
                                    transform_original.d,
                                    transform_original.e,
                                    transform_original.f + y * transform_original.e
                                )
                                perfil_actualizado['transform'] = nueva_transform
                            
                            # Definir la ruta de salida
                            nombre_bloque_paralela = f"bloque_{contador}_x{x}_y{y}.tiff"
                            ruta_salida_paralela = os.path.join(carpeta_salida_paralela, nombre_bloque_paralela)
                            
                            # Escribir el archivo multi-banda conservando el formato original
                            with rasterio.open(ruta_salida_paralela, 'w', **perfil_actualizado) as dst:
                                dst.write(datos_salida)
                            
                            paralelas_guardadas += 1
                            print(f"       Guardada imagen paralela con {len(bloques_paralelos)} bandas")
                    
                    contador += 1
                else:
                    descartadas += 1

        print(f"     Se guardaron {contador} bloques con color rojo detectado.")
        print(f"     Se guardaron {paralelas_guardadas} bloques paralelos.")
        print(f"     Se descartaron {descartadas} bloques sin rojo o incompletos.")
        
        if contador != paralelas_guardadas and imagen_paralela_data:
            print(f"     ⚠  ADVERTENCIA: Bloques principales ({contador}) no coinciden con paralelos ({paralelas_guardadas})")
        
        return contador
        
    except Exception as e:
        print(f"     Error procesando imagen: {str(e)}")
        import traceback
        print(f"     Detalles: {traceback.format_exc()}")
        return 0

# El resto del código se mantiene igual...
def encontrar_imagen_paralela(ruta_imagen, carpeta_actual):
    """
    Encuentra la imagen paralela para una imagen dada.
    """
    nombre_base = os.path.splitext(os.path.basename(ruta_imagen))[0]
    nombre_archivo = os.path.basename(ruta_imagen)
    
    print(f"     Buscando paralela para: {nombre_archivo}")
    
    # ✅ CORREGIDO: Si es NIR, buscar RGB
    if nombre_base.endswith("_NIR") or "NIR" in nombre_base.upper():
        nombre_paralelo = nombre_base.replace("_NIR", "_RGB")
        # Probar diferentes extensiones
        for ext in ['.tiff', '.tif']:
            posible_paralela = nombre_paralelo + ext
            if posible_paralela != nombre_archivo and os.path.exists(os.path.join(carpeta_actual, posible_paralela)):
                print(f"     ✅ Encontrada paralela RGB: {posible_paralela}")
                return os.path.join(carpeta_actual, posible_paralela)
    
    # ✅ CORREGIDO: Si es RGB, buscar NIR  
    elif nombre_base.endswith("_RGB") or "RGB" in nombre_base.upper():
        nombre_paralelo = nombre_base.replace("_RGB", "_NIR")
        # Probar diferentes extensiones
        for ext in ['.tiff', '.tif']:
            posible_paralela = nombre_paralelo + ext
            if posible_paralela != nombre_archivo and os.path.exists(os.path.join(carpeta_actual, posible_paralela)):
                print(f"     ✅ Encontrada paralela NIR: {posible_paralela}")
                return os.path.join(carpeta_actual, posible_paralela)
    
    # Si no encuentra por patrones, buscar cualquier archivo adicional
    archivos_posibles = []
    for archivo in os.listdir(carpeta_actual):
        if archivo.lower().endswith(('.tiff', '.tif')) and archivo != nombre_archivo:
            archivos_posibles.append(archivo)
    
    if len(archivos_posibles) == 1:
        print(f"     ✅ Encontrada única paralela: {archivos_posibles[0]}")
        return os.path.join(carpeta_actual, archivos_posibles[0])
    else:
        print(f"     ❌ No se encontró imagen paralela para: {nombre_archivo}")
        return None

def procesar_carpeta_raiz(carpeta_raiz, tamaño_bloque=128, umbral_rojo=100):
    """
    Recorre todas las subcarpetas de la carpeta raíz y procesa las imágenes TIFF satelitales
    """
    # Crear carpeta principal para todos los datasets
    carpeta_datasets = carpeta_raiz + "_datasets"
    os.makedirs(carpeta_datasets, exist_ok=True)
    
    print(f" Buscando imágenes TIFF satelitales en: {carpeta_raiz}")
    print(f" Los datasets se guardarán en: {carpeta_datasets}")
    print(" Configurado para 4 bandas satelitales (RGB + Infrarrojo)")
    print("  Píxeles rojos = ROJO UNIFORME (255,0,0) -  Todo lo demás en NEGRO")
    print("  Imágenes paralelas también serán recortadas y guardadas")
    print("-" * 60)
    
    total_imagenes = 0
    total_bloques = 0
    
    # Diccionario para llevar control de imágenes por subcarpeta
    contador_por_subcarpeta = {}
    
    # Recorrer todas las subcarpetas
    for root, dirs, files in os.walk(carpeta_raiz):
        # Filtrar solo archivos TIFF y ordenarlos para consistencia
        archivos_tiff = [f for f in files if f.lower().endswith(('.tiff', '.tif'))]
        
        if archivos_tiff:
            # Obtener identificador único para esta subcarpeta
            ruta_relativa = os.path.relpath(root, carpeta_raiz)
            if ruta_relativa == ".":
                id_subcarpeta = "raiz"
            else:
                id_subcarpeta = ruta_relativa.replace(os.sep, '_')
            
            # Inicializar contador para esta subcarpeta
            if id_subcarpeta not in contador_por_subcarpeta:
                contador_por_subcarpeta[id_subcarpeta] = 1
            
            print(f"\n Procesando subcarpeta: {ruta_relativa} ({len(archivos_tiff)} imágenes TIFF)")
            
            # Procesar cada imagen en esta subcarpeta
            for i, file in enumerate(sorted(archivos_tiff)):
                ruta_completa = os.path.join(root, file)
                
                # Buscar imagen paralela
                ruta_paralela = encontrar_imagen_paralela(ruta_completa, root)
                
                # Crear nombre único para la carpeta de salida (más corto)
                nombre_base = os.path.splitext(file)[0][:20]
                nombre_carpeta_salida = f"{id_subcarpeta}{contador_por_subcarpeta[id_subcarpeta]:02d}{nombre_base}"
                
                carpeta_salida_local = os.path.join(root, nombre_carpeta_salida)
                
                print(f"   Procesando imagen {i+1}/{len(archivos_tiff)}: {file}")
                if ruta_paralela:
                    print(f"   Imagen paralela encontrada: {os.path.basename(ruta_paralela)}")
                print(f"   Salida: {nombre_carpeta_salida}")
                
                try:
                    # Procesar la imagen con su paralela
                    bloques_generados = dividir_y_filtrar_imagen(
                        ruta_completa, 
                        ruta_paralela,
                        tamaño_bloque, 
                        carpeta_salida_local, 
                        umbral_rojo
                    )
                    
                    # Mover los bloques a la carpeta datasets centralizada
                    if bloques_generados > 0:
                        carpeta_destino = os.path.join(carpeta_datasets, nombre_carpeta_salida)
                        
                        if os.path.exists(carpeta_destino):
                            shutil.rmtree(carpeta_destino)
                        
                        shutil.move(carpeta_salida_local, carpeta_destino)
                        print(f"   Movidos {bloques_generados} bloques a dataset")
                    else:
                        if os.path.exists(carpeta_salida_local):
                            shutil.rmtree(carpeta_salida_local)
                        print("    No se generaron bloques, carpeta eliminada")
                    
                    total_imagenes += 1
                    total_bloques += bloques_generados
                    
                    contador_por_subcarpeta[id_subcarpeta] += 1
                    
                except Exception as e:
                    print(f"  Error procesando {file}: {str(e)}")
                    contador_por_subcarpeta[id_subcarpeta] += 1
    
    print("\n" + "=" * 60)
    print(f" PROCESO COMPLETADO")
    print(f" Total de imágenes procesadas: {total_imagenes}")
    print(f" Total de bloques generados: {total_bloques}")


if __name__ == "__main__":
    # --- CONFIGURACIÓN ---
    
    ruta_mis_imagenes = "/home/liese2/SPRI_AI_project/Dataset/Raw"
    
    # Tamaño del recorte
    tamano = 128
    
    # IMPORTANTE: Cambiado a 254. 
    # La lógica es "mayor que". Si pones 255, buscará valores > 255 (imposible).
    # Poniendo 254, detectará los píxeles que sean 255 (Rojo Puro).
    umbral = 254

    # --- VALIDACIÓN Y EJECUCIÓN ---
    if os.path.exists(ruta_mis_imagenes):
        print(f"Iniciando procesamiento en: {ruta_mis_imagenes}")
        
        # --- AQUÍ ESTABA EL ERROR: AHORA LA FUNCIÓN SE EJECUTA ---
        procesar_carpeta_raiz(ruta_mis_imagenes, tamano, umbral)
        
    else:
        print(f"Error: La ruta '{ruta_mis_imagenes}' no existe.")