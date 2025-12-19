"""Pascal VOC Dataset Segmentation Dataloader"""
import random

from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from torch.utils.data import Dataset
 
from PIL import Image
from torch.utils.data import DataLoader
from osgeo import gdal
import cv2

gdal.UseExceptions()

WILDFIRE_CLASSES = ('background',  # always index 0
               'wildfire' )

NUM_CLASSES = len(WILDFIRE_CLASSES) + 1

palettedata = [0,0,0,255,0,0, 0,255,0, 0,0,255, 255,255,0, 255,0,255, 0,255,255, 127,0,0, 0,127,0,  0,0,127, 127,127,0, 127,0,127, 0,127,127]
   
# Valores iniciales - serán recalculados
meanB = [0.0, 0.0, 0.0, 0.0]
stdB = [0.0, 0.0, 0.0, 0.0]

class WildFireDataset(Dataset):
    """Pascal VOC 2007 Dataset"""
    def __init__(self, list_file, img_dir, mask_dir, size=224, augmentation=False,transform=None,normalize=False,bands=4,niv=1):
        print("Init WildFireDataset size",size)
        self.images = open(list_file, "rt").read().split("\n")[:-1]
        self.transform = transform

        self.img_extension = ".tiff" 
        self.mask_extension = ".tiff"

        self.image_root_dir = img_dir
        self.mask_root_dir = mask_dir
        
        self.size=size
        print("Dataset size",size)
    
        # Forzamos siempre 4 bandas
        self.bands=4
        print("Dataset bands (forced to 4)", self.bands)
        
        self.normalize=normalize
        print("Dataset Normalize",normalize)
        
        self.augmentation=augmentation
        print("Dataset augmentation",augmentation)
        
        self.niv_aug=niv

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        name = self.images[index]
        image_path = os.path.join(self.image_root_dir, name + self.img_extension)
        mask_path = os.path.join(self.mask_root_dir, name + self.mask_extension)

        image = self.load_image(path=image_path)
        gt_mask = self.load_mask(path=mask_path)
        
        # Asegurar que la imagen tenga exactamente 4 bandas
        if image.shape[0] != 4:
            image = self._ensure_4_bands(image)
        
        if self.augmentation:
            if random.uniform(0, 1) > 0.25:  # 3/4 probabilidad
                ty = random.randint(0, 3)
                if ty == 0:
                    image = np.flip(image, axis=2)  # Flip horizontal
                    gt_mask = np.flip(gt_mask, axis=1)
                elif ty == 1:
                    image = np.flip(image, axis=1)  # Flip vertical
                    gt_mask = np.flip(gt_mask, axis=0)
                elif ty == 2:  # rot 90
                    image = np.rot90(image, 1, axes=(1, 2))
                    gt_mask = np.rot90(gt_mask, 1)
                elif ty == 3:  # rot -90
                    image = np.rot90(image, 3, axes=(1, 2))
                    gt_mask = np.rot90(gt_mask, 3)
            
            if self.niv_aug > 1:
                if self.niv_aug == 2:
                    max_b = 1
                    ecart = 0.05
                elif self.niv_aug == 3:
                    max_b = 3
                    ecart = 0.1
                elif self.niv_aug == 4:
                    ecart = 0.25
                    max_b = 4
                else:
                    max_b = 1
                    ecart = 0.05

                if random.uniform(0, 1) > 0.5:
                    if max_b == 1:
                        b = random.randint(0, 3)
                        delta = random.uniform(-ecart, ecart)
                        image[b, :, :] += delta
                        image[b, :, :] = np.clip(image[b, :, :], 0, 1)
                    else:
                        nb = random.randint(1, min(max_b, 4))
                        for _ in range(nb):
                            b = random.randint(0, 3)
                            delta = random.uniform(-ecart, ecart)
                            image[b, :, :] += delta
                            image[b, :, :] = np.clip(image[b, :, :], 0, 1)
 
        if self.normalize:
            for b in range(4):
                image[b, :, :] = (image[b, :, :] - meanB[b]) / stdB[b]
        
        data = {
            'image': torch.FloatTensor(image),
            'mask': torch.LongTensor(gt_mask)
        }
        
        return data

    def _ensure_4_bands(self, image):
        """Asegurar que la imagen tenga exactamente 4 bandas"""
        current_bands = image.shape[0]
        
        if current_bands == 4:
            return image
        elif current_bands == 3:
            # Añadir una cuarta banda de ceros
            fourth_band = np.zeros((1, self.size, self.size), dtype=np.float32)
            return np.concatenate([image, fourth_band], axis=0)
        elif current_bands < 3:
            # Caso raro: completar con ceros hasta 4 bandas
            zeros_to_add = 4 - current_bands
            extra_bands = np.zeros((zeros_to_add, self.size, self.size), dtype=np.float32)
            return np.concatenate([image, extra_bands], axis=0)
        elif current_bands > 4:
            # Tomar solo las primeras 4 bandas
            return image[:4, :, :]

    def load_image(self, path=None):
        """Cargar imagen y asegurar 4 bandas"""
        ds_img = gdal.Open(path)
        if not ds_img:
            print(f"Error opening image: {path}")
            # Crear imagen dummy de 4 bandas
            return np.zeros((4, self.size, self.size), dtype=np.float32)
        
        actual_bands = ds_img.RasterCount
        imx_t = np.empty([actual_bands, self.size, self.size], dtype=np.float32)
        
        for b in range(1, actual_bands + 1):
            channel = np.array(ds_img.GetRasterBand(b).ReadAsArray())
            
            # Redimensionar si es necesario
            if channel.shape[0] != self.size or channel.shape[1] != self.size:
                channel = cv2.resize(channel, (self.size, self.size))
            
            # Normalizar a [0, 1]
            if channel.dtype == np.uint8:
                np_float = channel.astype(np.float32) / 255.0
            elif channel.dtype == np.uint16:
                np_float = channel.astype(np.float32) / 65535.0
            else:
                np_float = channel.astype(np.float32)
                
            np_float = np.clip(np_float, 0.0, 1.0)
            imx_t[b-1, :, :] = np_float
        
        ds_img = None  # Liberar memoria
        
        # Asegurar 4 bandas antes de retornar
        return self._ensure_4_bands(imx_t)

    def load_mask(self, path=None):
        """Cargar máscara"""
        try:
            raw_image = Image.open(path).convert('L')
            raw_image = raw_image.resize((self.size, self.size), resample=Image.NEAREST)
            imx_t = np.array(raw_image)
            
            # Normalizar a valores binarios (0 o 1)
            imx_t = np.where(imx_t > 20, 1, 0)
            
            return imx_t
        except Exception as e:
            print(f"Error loading mask {path}: {e}")
            # Crear máscara dummy si hay error
            return np.zeros((self.size, self.size), dtype=np.int64)


def calculate_mean_std(dataset):
    """Calcular mean y std para exactamente 4 bandas"""
    print("Calculating mean and std for 4 bands...")
    
    mean = [0.0, 0.0, 0.0, 0.0]
    std = [0.0, 0.0, 0.0, 0.0]
    nb_samples = 0
    
    for i, data in enumerate(dataset):
        image = data['image'].cpu().numpy()
        
        # Asegurar que tengamos 4 bandas
        if image.shape[0] != 4:
            print(f"Warning: Image has {image.shape[0]} bands, forcing to 4")
            continue
        
        # Actualizar mean y std para cada banda
        for b in range(4):
            band_data = image[b, :, :]
            mean[b] += band_data.mean()
            std[b] += band_data.std()
        
        nb_samples += 1
        
        if i % 100 == 0:
            print(f"Processed {i} images...")
    
    if nb_samples == 0:
        print("No samples processed!")
        return mean, std
    
    # Calcular promedios
    mean = [m / nb_samples for m in mean]
    std = [s / nb_samples for s in std]
    
    print(f"\nTotal samples: {nb_samples}")
    print(f"Mean for 4 bands: {mean}")
    print(f"Std for 4 bands: {std}")
    
    return mean, std


if __name__ == "__main__":
    data_root = "/home/liese2/SPRI_AI_project/Wildfire" 
    list_file_path = os.path.join(data_root, "ImageSets", "Segmentation", "train.txt")
    img_dir = os.path.join(data_root, "Images")
    mask_dir = os.path.join(data_root, "SegmentationClass")

    print("Creating dataset (always 4 bands)...")
    objects_dataset = WildFireDataset(list_file=list_file_path,
                                      img_dir=img_dir,
                                      mask_dir=mask_dir,
                                      size=128,
                                      bands=4)  # Siempre 4 bandas
    
    # Calcular mean y std
    print("\n" + "="*60)
    print("CALCULATING MEAN AND STD FOR 4 BANDS")
    print("="*60)
    meanB, stdB = calculate_mean_std(objects_dataset)
    
    # Actualizar las variables globales
    globals()['meanB'] = meanB
    globals()['stdB'] = stdB
    
    print(f"\nFinal mean (4 bands): {meanB}")
    print(f"Final std (4 bands): {stdB}")
    
    # Guardar los resultados
    stats_file = os.path.join(data_root, "dataset_stats_4bands.txt")
    with open(stats_file, 'w') as f:
        f.write(f"Number of samples: {len(objects_dataset)}\n")
        f.write(f"Image size: 128x128\n")
        f.write(f"Bands: 4 (forced)\n")
        f.write(f"\nMean values:\n")
        for i, m in enumerate(meanB):
            f.write(f"  Band {i+1}: {m}\n")
        f.write(f"\nStd values:\n")
        for i, s in enumerate(stdB):
            f.write(f"  Band {i+1}: {s}\n")
    
    print(f"\nStatistics saved to: {stats_file}")
    
    # Probar con una muestra
    print("\n" + "="*60)
    print("TESTING WITH A SAMPLE")
    print("="*60)
    
    sample = objects_dataset[0]
    image, mask = sample['image'], sample['mask']
    
    print(f"Image shape: {image.shape}")  # Debería ser (4, 128, 128)
    print(f"Mask shape: {mask.shape}")    # Debería ser (128, 128)
    
    # Calcular estadísticas de la muestra
    image_np = image.numpy()
    print(f"\nSample statistics:")
    for b in range(4):
        band_data = image_np[b, :, :]
        print(f"  Band {b+1}: min={band_data.min():.4f}, max={band_data.max():.4f}, "
              f"mean={band_data.mean():.4f}, std={band_data.std():.4f}")
    
    # Visualización
    print("\nCreating visualization...")
    if image_np.shape[0] >= 3:
        # Tomar las primeras 3 bandas para RGB
        rgb_image = image_np[:3, :, :]
        rgb_image = np.transpose(rgb_image, (1, 2, 0))
        
        fig = plt.figure(figsize=(12, 5))
        
        a = fig.add_subplot(1, 2, 1)
        plt.imshow(rgb_image)
        plt.title(f"Image (4 bands forced)")
        plt.colorbar(label='Intensity')
        
        a = fig.add_subplot(1, 2, 2)
        mask_plot = plt.imshow(mask.numpy(), cmap='gray')
        plt.title("Mask (binary)")
        plt.colorbar(mask_plot, label='Class')
        
        plt.tight_layout()
        
        # Guardar la visualización
        vis_file = os.path.join(data_root, "sample_visualization.png")
        plt.savefig(vis_file, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {vis_file}")
        
        plt.show()
    
    print("\n" + "="*60)
    print("PROCESS COMPLETED")
    print("="*60)