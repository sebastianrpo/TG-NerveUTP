import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from model import UNet
from hyperbolic_model import HyperbolicUNet
from dataset import get_dataloaders
from utils import calculate_metrics

# --- Configuración ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Verifica que esta sea tu ruta correcta
DATA_DIR = r'C:\Users\Sebas\Documents\Dataset'

def evaluate_model(model, loader):
    """Evalúa un modelo en el loader proporcionado y retorna las métricas promedio."""
    model.eval()
    total_dice, total_iou, total_sens, total_spec = 0, 0, 0, 0
    loop = tqdm(loader, desc="Evaluando")

    with torch.no_grad():
        for data, targets in loop:
            data = data.to(device=DEVICE)
            targets = targets.float().to(device=DEVICE)

            predictions = model(data)
            
            dice, iou, sens, spec = calculate_metrics(predictions, targets)
            total_dice += dice
            total_iou += iou
            total_sens += sens
            total_spec += spec

    n_batches = len(loader)
    return total_dice / n_batches, total_iou / n_batches, total_sens / n_batches, total_spec / n_batches

def main():
    print("=> Preparando datos de prueba...")
    transform = A.Compose([A.Resize(height=256, width=256), ToTensorV2()])
    
    # Solo nos interesa el test_loader (el tercer valor que retorna la función)
    _, _, test_loader = get_dataloaders(DATA_DIR, transform, batch_size=8)
    
    # 1. Evaluar Baseline Euclidiano
    print("\n=> Cargando modelo Euclidiano (Baseline)...")
    model_base = UNet(n_channels=1, n_classes=1).to(DEVICE)
    try:
        model_base.load_state_dict(torch.load("unet_baseline_best.pth", map_location=DEVICE, weights_only=False)['state_dict'])
        dice_b, iou_b, sens_b, spec_b = evaluate_model(model_base, test_loader)
    except FileNotFoundError:
        print("No se encontró 'unet_baseline_best.pth'. Revisa el nombre del archivo.")
        return
        
    # 2. Evaluar Modelo Hiperbólico
    print("\n=> Cargando modelo Hiperbólico...")
    model_hyp = HyperbolicUNet(n_channels=1, n_classes=1, embed_dim=3, c=0.1).to(DEVICE)
    try:
        model_hyp.load_state_dict(torch.load("unet_hyperbolic_best.pth", map_location=DEVICE, weights_only=False)['state_dict'])
        dice_h, iou_h, sens_h, spec_h = evaluate_model(model_hyp, test_loader)
    except FileNotFoundError:
        print("No se encontró 'unet_hyperbolic_best.pth'. Revisa el nombre del archivo.")
        return
        
    # 3. Imprimir Tabla Comparativa
    print("\n" + "="*55)
    print("🏆 RESULTADOS FINALES EN EL SET DE PRUEBA 🏆")
    print("="*55)
    print(f"{'Métrica':<15} | {'U-Net Euclidiana':<18} | {'U-Net Hiperbólica'}")
    print("-" * 55)
    print(f"{'Dice':<15} | {dice_b:<18.4f} | {dice_h:.4f}")
    print(f"{'IoU':<15} | {iou_b:<18.4f} | {iou_h:.4f}")
    print(f"{'Sensibilidad':<15} | {sens_b:<18.4f} | {sens_h:.4f}")
    print(f"{'Especificidad':<15} | {spec_b:<18.4f} | {spec_h:.4f}")
    print("="*55)

if __name__ == '__main__':
    main()