import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
from krein_model import KreinUNet
from dataset import get_dataloaders

# --- Configuración ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_DIR = r'C:\Users\Sebas\Documents\Dataset'
#DATA_DIR = r'C:\Users\srestrepo01\Documents\TG\NerveUTP'
MODEL_PATH = "unet_krein_best.pth"

def main():
    print(f"=> Cargando modelo Pseudo-Euclidiano (Kreïn) desde {MODEL_PATH}...")
    
    # 1. Cargar la arquitectura y los pesos
    model = KreinUNet(n_channels=1, n_classes=1, dim_pos=128, dim_neg=64).to(DEVICE)
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint['state_dict'])
    except FileNotFoundError:
        print(f"¡No se encontró el archivo {MODEL_PATH}!")
        return
        
    model.eval()

    # 2. Hook para extraer las características antes del clasificador Kreïn
    embeddings = []
    def hook_fn(module, input, output):
        embeddings.append(output)
    
    # Nos enganchamos a la última convolución del decoder
    hook_handle = model.conv4.register_forward_hook(hook_fn)

    # 3. Cargar el DataLoader
    transform = A.Compose([A.Resize(height=256, width=256), ToTensorV2()])
    _, _, test_loader = get_dataloaders(DATA_DIR, transform, batch_size=1)
    
    # Seleccionar una imagen aleatoria
    dataset = test_loader.dataset
    idx = random.randint(0, len(dataset) - 1)
    image_tensor, mask_tensor = dataset[idx]
    
    # 4. Realizar la predicción y extraer mapas de energía
    embeddings.clear()
    with torch.no_grad():
        img_input = image_tensor.unsqueeze(0).to(DEVICE)
        pred_logits = model(img_input)
        pred_prob = torch.sigmoid(pred_logits)
        pred_mask = (pred_prob > 0.5).float().cpu().squeeze().numpy()
        
        # --- EXTRACCIÓN DEL ESPACIO DE KREÏN ---
        features = embeddings[0] # Formato: [1, 64, Alto, Ancho]
        feat_perm = features.permute(0, 2, 3, 1) # Mover canales al final
        
        # Pasar las características por ambos subespacios (RBF y Polinomial)
        phi_pos = model.krein_out.rff_pos(feat_perm).permute(0, 3, 1, 2)
        phi_neg = model.krein_out.rff_neg(feat_perm).permute(0, 3, 1, 2)
        
        # Calcular los mapas de energía (Logits parciales) a nivel de píxel
        logit_pos = torch.sum(model.krein_out.w_pos * phi_pos, dim=1).squeeze().cpu().numpy()
        logit_neg = torch.sum(model.krein_out.w_neg * phi_neg, dim=1).squeeze().cpu().numpy()
        
    hook_handle.remove()
        
    # 5. Preparar imágenes para OpenCV (Truco de auto-escalado Min-Max)
    img_show = image_tensor.squeeze().cpu().numpy()
    img_show = (img_show - img_show.min()) / (img_show.max() - img_show.min() + 1e-5)
    img_show = (img_show * 255).astype(np.uint8)
    img_color = cv2.cvtColor(img_show, cv2.COLOR_GRAY2RGB) 
    
    true_mask = (mask_tensor.squeeze().numpy() * 255).astype(np.uint8)
    pred_mask_uint8 = (pred_mask * 255).astype(np.uint8)
    
    # Dibujar contornos
    contours_pred, _ = cv2.findContours(pred_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_color, contours_pred, -1, (255, 0, 0), 2) # Rojo: Predicción
    
    contours_true, _ = cv2.findContours(true_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_color, contours_true, -1, (0, 255, 0), 2) # Verde: Real
    
    # 6. Desplegar la gráfica con 4 paneles
    fig, ax = plt.subplots(1, 4, figsize=(24, 6))
    
    ax[0].imshow(img_show, cmap='gray')
    ax[0].set_title('Ultrasonido Original')
    ax[0].axis('off')
    
    ax[1].imshow(img_color)
    ax[1].set_title('Predicción (Rojo) vs Real (Verde)')
    ax[1].axis('off')
    
    im_pos = ax[2].imshow(logit_pos, cmap='magma')
    ax[2].set_title('Energía Positiva (RBF)\nCerteza de ser Nervio')
    ax[2].axis('off')
    fig.colorbar(im_pos, ax=ax[2], fraction=0.046, pad=0.04)
    
    im_neg = ax[3].imshow(logit_neg, cmap='magma')
    ax[3].set_title('Energía Negativa (Polinomial)\nRepulsión al Ruido/Fondo')
    ax[3].axis('off')
    fig.colorbar(im_neg, ax=ax[3], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()