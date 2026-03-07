import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import albumentations as A
from albumentations.pytorch import ToTensorV2


class NerveUTPDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
   
        all_files = os.listdir(data_dir)
        self.image_files = [
            f for f in all_files 
            if not f.endswith('_mask.png') and f.endswith('.png')
        ]
        self.image_files.sort()
        #self.image_files = self.image_files[:50] # ¡Solo usamos 50 imágenes para probar! ¡¡¡¡TEMPORAL!!!!

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        base_name = img_name.replace('.png', '')
        mask_name = f"{base_name}_mask.png"
        
        img_path = os.path.join(self.data_dir, img_name)
        mask_path = os.path.join(self.data_dir, mask_name)

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            raise ValueError(f"No se pudo cargar: {img_name}")

        # Preprocesamiento base: Reducción de ruido speckle
        image = cv2.medianBlur(image, 5) 

        # Normalización a rango [0.0, 1.0]
        image = image.astype(np.float32) / 255.0
        
        # Binarización de la máscara
        mask = (mask > 0).astype(np.float32)

        # Aplicar transformaciones de Albumentations
        if self.transform:
            # Albumentations requiere que pasemos imagen y máscara juntas
            augmented = self.transform(image=image, mask=mask)
            image_tensor = augmented['image']
            mask_tensor = augmented['mask']
            
            # Ajuste de dimensiones: ToTensorV2 no le pone la dimensión de canal a la máscara si es 2D
            if mask_tensor.ndim == 2:
                mask_tensor = mask_tensor.unsqueeze(0)
        else:
            # Fallback en caso de no usar transformaciones
            image_tensor = torch.from_numpy(image).unsqueeze(0) 
            mask_tensor = torch.from_numpy(mask).unsqueeze(0)

        return image_tensor, mask_tensor

# --- Implementación y Transformaciones ---
# Definimos el pipeline de transformaciones
train_transform = A.Compose([
    A.Resize(height=256, width=256),  # Redimensionado a formato cuadrado estándar
    ToTensorV2()                      # Conversión final a tensor de PyTorch
])

def get_dataloaders(data_dir, transform, batch_size=8):
    """
    Divide el dataset en Entrenamiento (70%), Validación (15%) y Prueba (15%)
    y retorna sus respectivos DataLoaders listos para iterar.
    """
    # 1. Instanciar el dataset completo
    full_dataset = NerveUTPDataset(data_dir=data_dir, transform=transform)
    
    # 2. Calcular los tamaños exactos
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    # 3. Dividir aleatoriamente usando una semilla (seed) para reproducibilidad
    # Esto asegura que siempre se seleccionen las mismas imágenes para cada set
    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds, test_ds = random_split(
        full_dataset, 
        [train_size, val_size, test_size], 
        generator=generator
    )
    
    # 4. Crear los DataLoaders (solo se mezcla el de entrenamiento)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, test_loader

#data_folder = r'C:\Users\srestrepo01\Documents\TG\NerveUTP' #PC TRABAJO
data_folder = r'C:\Users\Sebas\Documents\Dataset'  #PC PERSONAL

# Instanciamos el dataset con las transformaciones
nerve_dataset = NerveUTPDataset(data_dir=data_folder, transform=train_transform)
train_loader = DataLoader(nerve_dataset, batch_size=8, shuffle=True, num_workers=0)

if __name__ == '__main__':
    #data_folder = r'C:\Users\srestrepo01\Documents\TG\NerveUTP' #PC TRABAJO
    data_folder = r'C:\Users\Sebas\Documents\Dataset'            #PC PERSONAL
    # Llamamos a la nueva función
    train_loader, val_loader, test_loader = get_dataloaders(data_folder, train_transform)
    
    print(f"Lotes de Entrenamiento: {len(train_loader)} (aprox. {len(train_loader.dataset)} imágenes)")
    print(f"Lotes de Validación: {len(val_loader)} (aprox. {len(val_loader.dataset)} imágenes)")
    print(f"Lotes de Prueba: {len(test_loader)} (aprox. {len(test_loader.dataset)} imágenes)")