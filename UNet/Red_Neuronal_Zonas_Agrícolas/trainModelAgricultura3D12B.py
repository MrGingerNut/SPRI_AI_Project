#python trainModelAgricultura3D12B.py --data C:\Users\Nitro\Downloads\Red_neuronal\cnn_agricultura\sourcedata\data --lr 0.001 -b 4 --epochs 100 --gpu False --image-size 256

import argparse
import os
import tempfile
import shutil
import time
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import Dataset
from osgeo import gdal
from agricultura_models import UNet2D as AgriculturaNet
import cv2
from PIL import Image

# Configuración
parser = argparse.ArgumentParser(description='Agricultural Segmentation Training')
parser.add_argument('--data', required=True, help='path to dataset')
parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=2, type=int, help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, help='initial learning rate')
parser.add_argument('--gpu', default="False", type=str, help='use GPU')
parser.add_argument('--image-size', default=256, type=int, help='target image size for resizing')
parser.add_argument('--print-freq', default=10, type=int, help='print frequency')

best_iou = 0

def custom_collate_fn(batch):
    return batch

class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, target_size=256):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.target_size = target_size
        # Solo imágenes TIFF
        self.images = [f for f in os.listdir(images_dir) 
                      if f.lower().endswith(('.tif', '.tiff'))]
        
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.images_dir, img_name)
        
        # ✅ CORREGIDO: SIEMPRE BUSCAR MÁSCARAS PNG
        base_name = os.path.splitext(img_name)[0]
        mask_name = base_name + '.png'  # Cambiar a .png
        mask_path = os.path.join(self.masks_dir, mask_name)
        
        # Verificar si la máscara existe
        if not os.path.exists(mask_path):
            print(f"ERROR: No se encontró máscara PNG para {img_name}")
            print(f"Buscando en: {mask_path}")
            # Retornar tensores vacíos como fallback
            return torch.zeros((4, self.target_size, self.target_size)), torch.zeros((self.target_size, self.target_size), dtype=torch.long)
        
        try:
            # Cargar imagen (con GDAL)
            img = self.load_and_resize_image(img_path, self.target_size)
            # Cargar máscara (con OpenCV/PIL)
            mask = self.load_and_resize_mask(mask_path, self.target_size)
                
            return img, mask
        except Exception as e:
            print(f"Error cargando {img_name}: {e}")
            return torch.zeros((4, self.target_size, self.target_size)), torch.zeros((self.target_size, self.target_size), dtype=torch.long)
    
    def load_and_resize_image(self, path, target_size):
        """Cargar imagen TIFF multibanda con GDAL"""
        ds = gdal.Open(path)
        if ds is None:
            raise ValueError(f"No se pudo abrir la imagen: {path}")
            
        bands = ds.RasterCount
        height, width = ds.RasterYSize, ds.RasterXSize
        
        # Si ya tiene el tamaño correcto
        if height == target_size and width == target_size:
            image = np.zeros((bands, target_size, target_size), dtype=np.float32)
            for b in range(bands):
                band_data = ds.GetRasterBand(b+1).ReadAsArray()
                band_data = band_data.astype(np.float32) / 12500.0
                band_data = np.clip(band_data, 0, 1)
                image[b, :, :] = band_data
            ds = None
            return torch.tensor(image, dtype=torch.float32)
        
        # Redimensionar con GDAL
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            gdal.Warp(temp_path, ds, width=target_size, height=target_size, 
                     resampleAlg=gdal.GRIORA_Bilinear, format='GTiff')
            
            ds_resized = gdal.Open(temp_path)
            image = np.zeros((bands, target_size, target_size), dtype=np.float32)
            
            for b in range(bands):
                band_data = ds_resized.GetRasterBand(b+1).ReadAsArray()
                band_data = band_data.astype(np.float32) / 12500.0
                band_data = np.clip(band_data, 0, 1)
                image[b, :, :] = band_data
            
            ds_resized = None
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
        
        ds = None
        return torch.tensor(image, dtype=torch.float32)
    
    def load_and_resize_mask(self, path, target_size):
        """Cargar máscara PNG y asegurar que sea binaria (0=no agrícola, 1=agrícola)"""
        try:
            # Cargar con OpenCV
            mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                # Si OpenCV falla, intentar con PIL
                mask_img = Image.open(path)
                mask = np.array(mask_img)
                if len(mask.shape) == 3:  # Si es RGB, convertir a escala de grises
                    mask = mask[:, :, 0]
                    
            # Redimensionar si es necesario
            if mask.shape != (target_size, target_size):
                mask = cv2.resize(mask, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
            
            # ✅ CORREGIDO: Binarizar correctamente
            # Asumir que cualquier valor > 0 es agrícola
            mask_binary = np.where(mask > 0, 1, 0).astype(np.uint8)
            
            mask_tensor = torch.tensor(mask_binary, dtype=torch.long)
            
            return mask_tensor
        except Exception as e:
            print(f"Error cargando máscara {path}: {e}")
            # Retornar máscara vacía (solo fondo)
            return torch.zeros((target_size, target_size), dtype=torch.long)

def calculate_iou(pred, target):
    """Calculate Intersection over Union para segmentación binaria (solo clase agrícola)"""
    # pred: [batch, classes, height, width]
    # target: [batch, height, width] - solo 0s y 1s
    
    # Obtener la clase predicha para cada pixel (usamos softmax + argmax)
    pred = torch.softmax(pred, dim=1)  # Aplicar softmax primero
    pred = torch.argmax(pred, dim=1)   # [batch, height, width]
    
    # Asegurar que las dimensiones coincidan
    if pred.shape != target.shape:
        min_height = min(pred.shape[1], target.shape[1])
        min_width = min(pred.shape[2], target.shape[2])
        pred = pred[:, :min_height, :min_width]
        target = target[:, :min_height, :min_width]
    
    # Calcular IoU para clase 1 (agrícola)
    pred_agricola = pred == 1
    target_agricola = target == 1
    
    intersection = (pred_agricola & target_agricola).float().sum()
    union = (pred_agricola | target_agricola).float().sum()
    
    if union == 0:
        return 0.0
    else:
        return (intersection / union).item()

def check_class_balance(dataset, name="Dataset"):
    """Verificar el balance entre clases agrícola vs no agrícola"""
    total_pixels = 0
    agricola_pixels = 0
    
    for i in range(min(10, len(dataset))):  # Revisar solo las primeras 10 imágenes
        _, mask = dataset[i]
        total_pixels += mask.numel()
        agricola_pixels += (mask == 1).sum().item()
    
    if total_pixels > 0:
        agricola_percent = (agricola_pixels / total_pixels) * 100
        print(f"{name} - Píxeles agrícolas: {agricola_percent:.2f}%")
    else:
        print(f"{name} - No se pudieron contar píxeles")

def process_batch_train(model, batch, criterion, gpu, optimizer):
    """Procesar un batch para entrenamiento"""
    batch_loss = 0
    batch_iou = 0
    n_samples = 0
    
    for i, (input, target) in enumerate(batch):
        if gpu:
            input = input.cuda()
            target = target.cuda()
        
        # Asegurar que input tenga 4 dimensiones [batch, channels, height, width]
        if input.dim() == 3:
            input = input.unsqueeze(0)  # [1, channels, height, width]
        
        # Asegurar que target tenga 3 dimensiones [batch, height, width]  
        if target.dim() == 2:
            target = target.unsqueeze(0)  # [1, height, width]
        
        # Forward
        output = model(input)
        
        # Calcular pérdida
        loss = criterion(output, target)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        batch_loss += loss.item()
        
        # Calcular IoU
        with torch.no_grad():
            iou = calculate_iou(output, target)
            if not np.isnan(iou):
                batch_iou += iou
        
        n_samples += 1
    
    return batch_loss / n_samples, batch_iou / n_samples if n_samples > 0 else 0

def process_batch_validate(model, batch, gpu):
    """Procesar un batch para validación (sin calcular pérdida)"""
    batch_iou = 0
    n_samples = 0
    
    for i, (input, target) in enumerate(batch):
        if gpu:
            input = input.cuda()
            target = target.cuda()
        
        # Asegurar que input tenga 4 dimensiones [batch, channels, height, width]
        if input.dim() == 3:
            input = input.unsqueeze(0)  # [1, channels, height, width]
        
        # Asegurar que target tenga 3 dimensiones [batch, height, width]  
        if target.dim() == 2:
            target = target.unsqueeze(0)  # [1, height, width]
        
        # Forward
        with torch.no_grad():
            output = model(input)
            
            # Calcular IoU
            iou = calculate_iou(output, target)
            if not np.isnan(iou):
                batch_iou += iou
        
        n_samples += 1
    
    return batch_iou / n_samples if n_samples > 0 else 0

def train(train_loader, model, criterion, optimizer, epoch, gpu):
    model.train()
    losses = AverageMeter()
    ious = AverageMeter()
    
    for i, batch in enumerate(train_loader):
        batch_loss, batch_iou = process_batch_train(model, batch, criterion, gpu, optimizer)
        losses.update(batch_loss, len(batch))
        ious.update(batch_iou, len(batch))
        
        if i % args.print_freq == 0:
            print(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\tLoss: {losses.avg:.4f}\tIOU: {ious.avg:.4f}')
    
    return losses.avg

def validate(val_loader, model, epoch, gpu):
    model.eval()
    ious = AverageMeter()
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            batch_iou = process_batch_validate(model, batch, gpu)
            ious.update(batch_iou, len(batch))
            
            if i % args.print_freq == 0:
                print(f'Val: [{epoch}][{i}/{len(val_loader)}]\tIOU: {batch_iou:.4f}')
    
    return ious.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(os.path.dirname(filename), 'model_best.pth')
        shutil.copyfile(filename, best_filename)

class AverageMeter(object):
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def main():
    global args, best_iou
    args = parser.parse_args()
    gpu = args.gpu == "True"
    
    print("Usando GPU:", gpu)
    print("Tamaño objetivo de imagen:", args.image_size)
    
    # Crear directorio de pesos
    weights_dir = 'weights'
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)
    
    # Directorios de datos
    train_img_dir = os.path.join(args.data, 'train', 'images')
    train_mask_dir = os.path.join(args.data, 'train', 'masks')
    val_img_dir = os.path.join(args.data, 'val', 'images')
    val_mask_dir = os.path.join(args.data, 'val', 'masks')
    
    # Verificar directorios
    for dir_path in [train_img_dir, train_mask_dir, val_img_dir, val_mask_dir]:
        if not os.path.exists(dir_path):
            print(f"ERROR: No existe el directorio: {dir_path}")
            return
    
    # Crear datasets
    train_dataset = SegmentationDataset(train_img_dir, train_mask_dir, args.image_size)
    val_dataset = SegmentationDataset(val_img_dir, val_mask_dir, args.image_size)
    
    print(f"Imágenes de entrenamiento: {len(train_dataset)}")
    print(f"Imágenes de validación: {len(val_dataset)}")
    
    # ✅ Verificar balance de clases
    check_class_balance(train_dataset, "Entrenamiento")
    check_class_balance(val_dataset, "Validación")
    
    # DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=0,
        collate_fn=custom_collate_fn,
        pin_memory=False)
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=custom_collate_fn,
        pin_memory=False)
    
    # Modelo
    model = AgriculturaNet(in_channels=4, out_channels=2)  # 2 clases: no agrícola (0) y agrícola (1)
    
    if gpu:
        model.cuda()
        print("Modelo en GPU")
    else:
        print("Modelo en CPU")
    
    # ✅ MODIFICADO: Usar pérdida con pesos de clase para manejar desbalance
    # Si hay mucho más fondo que zonas agrícolas, dar más peso a la clase agrícola
    # Puedes ajustar estos pesos según el balance de tus clases
    class_weights = torch.tensor([1.0, 3.0])  # [peso_fondo, peso_agricola]
    if gpu:
        class_weights = class_weights.cuda()
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Archivos para guardar resultados
    log_file = open('training_log.txt', 'w')
    iou_file = open('iou_metrics.txt', 'w')
    log_file.write("Epoch,Loss,IOU_Train,IOU_Val,Time\n")
    iou_file.write("Epoch,IOU_Train,IOU_Val\n")
    
    # Entrenamiento
    for epoch in range(args.epochs):
        start_time = time.time()
        
        train_loss = train(train_loader, model, criterion, optimizer, epoch, gpu)
        val_iou = validate(val_loader, model, epoch, gpu)
        
        epoch_time = time.time() - start_time
        
        # Guardar logs
        log_file.write(f"{epoch},{train_loss:.6f},{0:.4f},{val_iou:.4f},{epoch_time:.2f}\n")
        iou_file.write(f"{epoch},{0:.4f},{val_iou:.4f}\n")
        log_file.flush()
        iou_file.flush()
        
        print(f'Epoch {epoch}: Loss: {train_loss:.4f}, IOU Val: {val_iou:.4f}, Time: {epoch_time:.2f}s')
        
        # Guardar checkpoint
        is_best = val_iou > best_iou
        best_iou = max(val_iou, best_iou)
        
        if is_best:
            torch.save(model.state_dict(), os.path.join(weights_dir, 'model_best.pth'))
            print(f"Mejor modelo guardado con IOU: {best_iou:.4f}")
        
        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(weights_dir, f'model_epoch_{epoch}.pth'))
    
    log_file.close()
    iou_file.close()
    torch.save(model.state_dict(), os.path.join(weights_dir, 'model_final.pth'))
    print("Entrenamiento completado!")

if __name__ == '__main__':
    main()