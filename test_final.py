import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from model import UNet
from hyperbolic_model import HyperbolicUNet
from krein_model import KreinUNet
from dataset import get_dataloaders
from utils import calculate_metrics

# --- Configuración ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Ajusta esta ruta si lo vas a correr en Google Colab
DATA_DIR = r'C:\Users\Sebas\Documents\Dataset'
#DATA_DIR = r'C:\Users\srestrepo01\Documents\TG\NerveUTP' 

def evaluate_model(model, loader, model_name):
    """Pasa todo el Test Set por el modelo y calcula los promedios."""
    model.eval()
    total_dice, total_iou, total_sens, total_spec = 0, 0, 0, 0
    loop = tqdm(loader, desc=f"Evaluando {model_name}")

    with torch.no_grad():
        for data, targets in loop:
            data = data.to(device=DEVICE)
            targets = targets.float().to(device=DEVICE)

            # Las 3 redes están configuradas para retornar solo los logits en modo eval()
            predictions = model(data)
            
            dice, iou, sens, spec = calculate_metrics(predictions, targets)
            total_dice += dice
            total_iou += iou
            total_sens += sens
            total_spec += spec

    n_batches = len(loader)
    return total_dice / n_batches, total_iou / n_batches, total_sens / n_batches, total_spec / n_batches

def main():
    print("=> Preparando datos de prueba (Test Set 15% inédito)...")
    transform = A.Compose([A.Resize(height=256, width=256), ToTensorV2()])
    _, _, test_loader = get_dataloaders(DATA_DIR, transform, batch_size=8)
    
    resultados = {}

    # 1. Evaluar U-Net Clásica (Euclidiana)
    print("\n=> 1. Cargando U-Net Clásica (Euclidiana)...")
    model_base = UNet(n_channels=1, n_classes=1).to(DEVICE)
    try:
        checkpoint = torch.load("unet_baseline_best.pth", map_location=DEVICE, weights_only=False)
        model_base.load_state_dict(checkpoint['state_dict'])
        resultados['Euclidiana'] = evaluate_model(model_base, test_loader, "Euclidiano")
    except FileNotFoundError:
        print("  [!] Archivo 'unet_baseline_best.pth' no encontrado. Saltando...")

    # 2. Evaluar U-Net Hiperbólica
    print("\n=> 2. Cargando U-Net Hiperbólica...")
    model_hyp = HyperbolicUNet(n_channels=1, n_classes=1, embed_dim=3, c=0.1).to(DEVICE)
    try:
        checkpoint = torch.load("unet_hyperbolic_best.pth", map_location=DEVICE, weights_only=False)
        model_hyp.load_state_dict(checkpoint['state_dict'])
        resultados['Hiperbólica'] = evaluate_model(model_hyp, test_loader, "Hiperbólico")
    except FileNotFoundError:
        print("  [!] Archivo 'unet_hyperbolic_best.pth' no encontrado. Saltando...")

    # 3. Evaluar U-Net Pseudo-Euclidiana (Kreïn)
    print("\n=> 3. Cargando U-Net Pseudo-Euclidiana (Kreïn)...")
    model_krein = KreinUNet(n_channels=1, n_classes=1, dim_pos=128, dim_neg=64).to(DEVICE)
    try:
        checkpoint = torch.load("unet_krein_best.pth", map_location=DEVICE, weights_only=False)
        model_krein.load_state_dict(checkpoint['state_dict'])
        resultados['Kreïn'] = evaluate_model(model_krein, test_loader, "Kreïn")
    except FileNotFoundError:
        print("  [!] Archivo 'unet_krein_best.pth' no encontrado. Saltando...")

    # Imprimir Tabla Comparativa Definitiva
    print("\n" + "="*75)
    print("🏆 RESULTADOS FINALES EN EL SET DE PRUEBA 🏆")
    print("="*75)
    print(f"{'Modelo':<20} | {'Dice':<10} | {'IoU':<10} | {'Sensibilidad':<12} | {'Especificidad':<10}")
    print("-" * 75)
    
    for nombre, metricas in resultados.items():
        dice, iou, sens, spec = metricas
        print(f"{nombre:<20} | {dice:<10.4f} | {iou:<10.4f} | {sens:<12.4f} | {spec:<10.4f}")
    print("="*75)

if __name__ == '__main__':
    main()