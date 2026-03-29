import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Importamos nuestros módulos
from model import UNet
from dataset import get_dataloaders

# --- Configuración ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#DATA_DIR = r'C:\Users\srestrepo01\Documents\TG\NerveUTP'
DATA_DIR = r'C:\Users\Sebas\Documents\Dataset'

MODEL_PATH = "unet_baseline_best.pth"

def main():
    print(f"=> Cargando modelo desde {MODEL_PATH}...")
    
    # 1. Cargar la arquitectura y los pesos entrenados
    model = UNet(n_channels=1, n_classes=1).to(DEVICE)
    try:
        # weights_only=True es una buena práctica de seguridad en PyTorch moderno
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint['state_dict'])
    except FileNotFoundError:
        print("¡El modelo aún no se ha guardado! Espera a que termine la primera época de entrenamiento.")
        return
        
    model.eval()

    # 2. Cargar el DataLoader (usamos el pipeline completo para tener el set de prueba)
    transform = A.Compose([A.Resize(height=256, width=256), ToTensorV2()])
    _, _, test_loader = get_dataloaders(DATA_DIR, transform, batch_size=1)
    
    # 3. Seleccionar una imagen aleatoria del set de prueba
    dataset = test_loader.dataset
    idx = random.randint(0, len(dataset) - 1)
    image_tensor, mask_tensor, img_name = dataset[idx]
    
    # 4. Realizar la predicción
    with torch.no_grad():
        img_input = image_tensor.unsqueeze(0).to(DEVICE) # Añadir dimensión de batch [1, 1, 256, 256]
        pred_logits = model(img_input)
        pred_prob = torch.sigmoid(pred_logits)
        # Binarizar la predicción usando un umbral de 0.5
        pred_mask = (pred_prob > 0.5).float().cpu().squeeze().numpy()
        
    # 5. Preparar los tensores para visualización (convertir a formato de imagen NumPy)
    img_show = image_tensor.squeeze().numpy()
    img_show = (img_show * 255).astype(np.uint8) # Escala 0-255
    
    # Crear una versión a color para poder dibujar las líneas rojas y verdes
    img_color = cv2.cvtColor(img_show, cv2.COLOR_GRAY2RGB) 
    true_mask = mask_tensor.squeeze().numpy().astype(np.uint8)
    pred_mask_uint8 = pred_mask.astype(np.uint8)
    
    # 6. Dibujar los contornos usando OpenCV
    # Contorno PREDICHO en ROJO
    contours_pred, _ = cv2.findContours(pred_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_color, contours_pred, -1, (255, 0, 0), 2) # (R, G, B) en matplotlib
    
    # Contorno REAL (Ground Truth) en VERDE para comparar
    contours_true, _ = cv2.findContours(true_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_color, contours_true, -1, (0, 255, 0), 2)
    
    # 7. Desplegar la gráfica
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    
    ax[0].imshow(img_show, cmap='gray')
    ax[0].set_title('Ultrasonido Original')
    ax[0].axis('off')
    
    ax[1].imshow(true_mask, cmap='gray')
    ax[1].set_title('Máscara Real (Ground Truth)')
    ax[1].axis('off')
    
    ax[2].imshow(img_color)
    ax[2].set_title('Predicción (Rojo) vs Real (Verde)')
    ax[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    print(f"[*] Visualizando imagen: {img_name}")

if __name__ == '__main__':
    main()