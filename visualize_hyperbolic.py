import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
from hyperbolic_model import HyperbolicUNet, expmap0
from dataset import get_dataloaders

# --- Configuración ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#DATA_DIR = r'C:\Users\srestrepo01\Documents\TG\NerveUTP' 
DATA_DIR = r'C:\Users\Sebas\Documents\Dataset'
MODEL_PATH = "unet_hyperbolic_best.pth"

def main():
    print(f"=> Cargando modelo hiperbólico desde {MODEL_PATH}...")
    
    #Cargar la arquitectura y los pesos entrenados
    model = HyperbolicUNet(n_channels=1, n_classes=1, embed_dim=3, c=0.1).to(DEVICE)
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint['state_dict'])
    except FileNotFoundError:
        print(f"¡No se encontró el archivo {MODEL_PATH}!")
        return
        
    model.eval()

    #Extraer los embebidos latentes usando un Hook
    # Esto nos permite obtener el mapa de incertidumbre
    embeddings = []
    def hook_fn(module, input, output):
        embeddings.append(output)
    
    # Nos enganchamos a la capa que reduce la dimensionalidad (justo antes del clasificador)
    hook_handle = model.to_embed_dim.register_forward_hook(hook_fn)

    #Cargar el DataLoader (usamos el pipeline completo para tener el set de prueba)
    transform = A.Compose([A.Resize(height=256, width=256), ToTensorV2()])
    _, _, test_loader = get_dataloaders(DATA_DIR, transform, batch_size=1)
    
    #Seleccionar una imagen aleatoria del set de prueba
    dataset = test_loader.dataset
    idx = random.randint(0, len(dataset) - 1)
    image_tensor, mask_tensor = dataset[idx]
    
    #Realizar la predicción
    embeddings.clear() # Limpiar embeddings de corridas anteriores
    with torch.no_grad():
        img_input = image_tensor.unsqueeze(0).to(DEVICE) 
        pred_logits = model(img_input)
        pred_prob = torch.sigmoid(pred_logits)
        pred_mask = (pred_prob > 0.5).float().cpu().squeeze().numpy()
        
        # Obtener el mapa de incertidumbre/confianza (Norma L2 en la bola de Poincaré)
        latent_features = embeddings[0] # Formato: [1, embed_dim, Alto, Ancho]
        z_poincare = expmap0(latent_features, model.hyperbolic_out.c)
        # Calculamos la distancia (norma L2) por cada píxel a través de los canales
        confidence_map = torch.norm(z_poincare, p=2, dim=1).cpu().squeeze().numpy()
        
    # Limpiamos el hook
    hook_handle.remove()
        
    # 5. Preparar los tensores para visualización 
    img_show = image_tensor.squeeze().numpy()
    img_show = (img_show * 255).astype(np.uint8)
    img_color = cv2.cvtColor(img_show, cv2.COLOR_GRAY2RGB) 
    
    true_mask = mask_tensor.squeeze().numpy().astype(np.uint8)
    pred_mask_uint8 = pred_mask.astype(np.uint8)
    
    # 6. Dibujar los contornos usando OpenCV
    contours_pred, _ = cv2.findContours(pred_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_color, contours_pred, -1, (255, 0, 0), 2) # Rojo: Predicción
    
    contours_true, _ = cv2.findContours(true_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_color, contours_true, -1, (0, 255, 0), 2) # Verde: Real
    
    # 7. Desplegar la gráfica con el mapa de confianza
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    
    ax[0].imshow(img_show, cmap='gray')
    ax[0].set_title('Ultrasonido Original')
    ax[0].axis('off')
    
    ax[1].imshow(img_color)
    ax[1].set_title('Predicción (Rojo) vs Real (Verde)')
    ax[1].axis('off')
    
    # Mapa de calor de la confianza hiperbólica
    im = ax[2].imshow(confidence_map, cmap='magma')
    ax[2].set_title('Mapa de Confianza Hiperbólica')
    ax[2].axis('off')
    fig.colorbar(im, ax=ax[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()