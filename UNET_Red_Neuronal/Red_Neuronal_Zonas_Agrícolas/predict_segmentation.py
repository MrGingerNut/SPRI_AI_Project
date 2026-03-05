#python predict_segmentation.py --model weights/model_best.pth --image prueba.tif --output prediccion.tif --gpu False
#python predict_segmentation.py --model "C:\Users\Nitro\Downloads\Red_neuronal\cnn_agricultura\sourcedata\weights\model_best.pth" --image "C:\Users\Nitro\Downloads\Red_neuronal\cnn_agricultura\sourcedata\Detectar\ID_3_RGB.tif" --output "C:\Users\Nitro\Downloads\Red_neuronal\cnn_agricultura\sourcedata\Detectar\ID_3_RGB_prediccion.tif"

import torch
import numpy as np
from osgeo import gdal, osr
import os
from agricultura_models import UNet2D as AgriculturaNet
import cv2

def predict_image(model_path, image_path, output_path, gpu=False):
    # Cargar modelo
    model = AgriculturaNet(in_channels=4, out_channels=2)
    device = torch.device('cuda' if gpu else 'cpu')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Cargar imagen
    ds = gdal.Open(image_path)
    if ds is None:
        print(f"Error: No se pudo abrir la imagen {image_path}")
        return
    
    bands = ds.RasterCount
    height, width = ds.RasterYSize, ds.RasterXSize
    
    print(f"Imagen: {bands} bandas, {width}x{height} píxeles")
    
    # Preparar imagen
    image = np.zeros((bands, height, width), dtype=np.float32)
    for b in range(bands):
        band_data = ds.GetRasterBand(b+1).ReadAsArray()
        band_data = band_data.astype(np.float32) / 12500.0
        band_data = np.clip(band_data, 0, 1)
        image[b, :, :] = band_data
    
    # Predecir por patches si la imagen es muy grande
    patch_size = 256
    output = np.zeros((2, height, width), dtype=np.float32)
    count = np.zeros((height, width), dtype=np.float32)
    
    print("Procesando imagen por patches...")
    
    with torch.no_grad():
        for i in range(0, height, patch_size):
            for j in range(0, width, patch_size):
                i_end = min(i + patch_size, height)
                j_end = min(j + patch_size, width)
                
                # Obtener patch
                patch = image[:, i:i_end, j:j_end]
                
                # Si el patch es más pequeño que patch_size, rellenar
                if patch.shape[1] < patch_size or patch.shape[2] < patch_size:
                    temp_patch = np.zeros((bands, patch_size, patch_size), dtype=np.float32)
                    temp_patch[:, :patch.shape[1], :patch.shape[2]] = patch
                    patch = temp_patch
                
                patch_tensor = torch.tensor(patch, dtype=torch.float32).unsqueeze(0)
                
                if gpu:
                    patch_tensor = patch_tensor.cuda()
                
                # Predicción
                pred_patch = model(patch_tensor)
                pred_patch = torch.softmax(pred_patch, dim=1).cpu().numpy()[0]
                
                # Recortar si se rellenó
                if patch.shape[1] > (i_end - i) or patch.shape[2] > (j_end - j):
                    pred_patch = pred_patch[:, :(i_end - i), :(j_end - j)]
                
                # Acumular resultados
                output[:, i:i_end, j:j_end] += pred_patch
                count[i:i_end, j:j_end] += 1
    
    # Promediar superposiciones
    output = output / np.maximum(count, 1)  # Evitar división por cero
    
    # ✅ MODIFICADO: Usar softmax y umbral para clase agrícola
    # output[1] contiene las probabilidades de la clase agrícola
    prob_agricola = output[1]
    
    # Aplicar umbral (puedes ajustar este valor)
    umbral = 0.5
    pred_mask = (prob_agricola > umbral).astype(np.uint8) * 255
    
    # Estadísticas de la predicción
    total_pixels = pred_mask.size
    agricola_pixels = np.sum(pred_mask > 0)
    print(f"Predicción - Píxeles agrícolas detectados: {agricola_pixels}/{total_pixels} ({agricola_pixels/total_pixels*100:.2f}%)")
    
    # Guardar resultado
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(output_path, width, height, 1, gdal.GDT_Byte)
    
    # Copiar georeferencia
    out_ds.SetGeoTransform(ds.GetGeoTransform())
    out_ds.SetProjection(ds.GetProjection())
    
    # Guardar máscara de predicción (clase 1: agrícola)
    out_ds.GetRasterBand(1).WriteArray(pred_mask)
    out_ds.FlushCache()
    
    ds = None
    out_ds = None
    
    print(f"Predicción guardada en: {output_path}")
    
    # Opcional: Guardar también una versión visual con colores
    output_visual_path = output_path.replace('.tif', '_visual.png')
    visual_mask = np.zeros((height, width, 3), dtype=np.uint8)
    visual_mask[pred_mask == 255] = [0, 255, 0]  # Verde para zonas agrícolas
    cv2.imwrite(output_visual_path, visual_mask)
    print(f"Visualización guardada en: {output_visual_path}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to model weights')
    parser.add_argument('--image', required=True, help='Path to input image')
    parser.add_argument('--output', required=True, help='Path for output prediction')
    parser.add_argument('--gpu', type=str, default="False", help='Use GPU')
    
    args = parser.parse_args()
    predict_image(args.model, args.image, args.output, args.gpu == "True")