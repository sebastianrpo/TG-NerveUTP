import os
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Importamos las arquitecturas desde los archivos
from model import UNet
from hyperbolic_model import HyperbolicUNet
from krein_model import KreinUNet

# Importamos utilidades
from dataset import get_dataloaders
from utils import calculate_metrics

# --- Configuración ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Usamos tu ruta personal
DATA_DIR = r'C:\Users\Sebas\Documents\Dataset'
BATCH_SIZE = 8

def evaluar_modelo(nombre_modelo, modelo, ruta_pesos, dataloader):
    print(f"\n{'='*60}")
    print(f"EVALUANDO MODELO: {nombre_modelo}")
    print(f"{'='*60}")

    # Cargar pesos
    try:
        checkpoint = torch.load(ruta_pesos, map_location=DEVICE, weights_only=False)
        modelo.load_state_dict(checkpoint['state_dict'])
        print(f"[*] Pesos cargados correctamente desde {ruta_pesos}")
    except FileNotFoundError:
        print(f"[!] ERROR: No se encontró el archivo {ruta_pesos}. Saltando modelo...")
        return

    modelo = modelo.to(DEVICE)
    modelo.eval()

    # Diccionario para agrupar las métricas
    resultados = {
        'ciatico': {'dice': [], 'iou': [], 'sens': [], 'spec': []},
        'cubital': {'dice': [], 'iou': [], 'sens': [], 'spec': []},
        'mediano': {'dice': [], 'iou': [], 'sens': [], 'spec': []},
        'femoral': {'dice': [], 'iou': [], 'sens': [], 'spec': []}
    }
    
    global_metrics = {'dice': [], 'iou': [], 'sens': [], 'spec': []}

    # Ciclo de Inferencia
    with torch.no_grad():
        # Desempaquetamos los 3 valores que ahora retorna el dataloader
        for imagenes, mascaras_reales, nombres_archivos in dataloader:
            imagenes = imagenes.to(DEVICE)
            mascaras_reales = mascaras_reales.to(DEVICE)

            # Pasar por el modelo (Todos retornan logits en modo eval)
            logits = modelo(imagenes)

            # Procesar imagen por imagen dentro del batch
            for i in range(imagenes.size(0)):
                # Extraer la predicción y la máscara real individual
                logit_img = logits[i:i+1] # Slice para mantener la dimensión [1, C, H, W]
                real_img = mascaras_reales[i:i+1]
                nombre = nombres_archivos[i].lower()

                # Calcular métricas usando la función de utils.py
                dice_val, iou_val, sens_val, spec_val = calculate_metrics(logit_img, real_img)

                # Guardar globales
                global_metrics['dice'].append(dice_val)
                global_metrics['iou'].append(iou_val)
                global_metrics['sens'].append(sens_val)
                global_metrics['spec'].append(spec_val)

                # Clasificar por tipo de nervio
                if 'ciatico' in nombre:
                    nervio_key = 'ciatico'
                elif 'cubital' in nombre:
                    nervio_key = 'cubital'
                elif 'mediano' in nombre:
                    nervio_key = 'mediano'
                elif 'femoral' in nombre:
                    nervio_key = 'femoral'
                else:
                    continue # Si hay algún archivo con otro nombre, lo ignoramos

                resultados[nervio_key]['dice'].append(dice_val)
                resultados[nervio_key]['iou'].append(iou_val)
                resultados[nervio_key]['sens'].append(sens_val)
                resultados[nervio_key]['spec'].append(spec_val)

    # Imprimir el reporte detallado
    print("\n--- MÉTRICAS POR TIPO DE NERVIO ---")
    for nervio, metricas in resultados.items():
        n_imgs = len(metricas['dice'])
        if n_imgs == 0:
            print(f"{nervio.capitalize()}: No hay imágenes en el set de prueba.")
            continue
            
        p_dice = np.mean(metricas['dice'])
        p_iou = np.mean(metricas['iou'])
        p_sens = np.mean(metricas['sens'])
        p_spec = np.mean(metricas['spec'])
        
        print(f"• {nervio.capitalize()} (n={n_imgs}):")
        print(f"  Dice: {p_dice:.4f} | IoU: {p_iou:.4f} | Sens: {p_sens:.4f} | Spec: {p_spec:.4f}")

    print("\n--- MÉTRICAS GLOBALES DEL MODELO ---")
    print(f"  Dice: {np.mean(global_metrics['dice']):.4f} | IoU: {np.mean(global_metrics['iou']):.4f} | Sens: {np.mean(global_metrics['sens']):.4f} | Spec: {np.mean(global_metrics['spec']):.4f}\n")

def main():
    print(f"=> Iniciando evaluación en: {DEVICE.upper()}")
    
    # Preparamos el dataloader de prueba
    transform = A.Compose([
        A.Resize(height=256, width=256),
        ToTensorV2()
    ])
    
    _, _, test_loader = get_dataloaders(DATA_DIR, transform, BATCH_SIZE)
    print(f"[*] Imágenes en el set de prueba: {len(test_loader.dataset)}")

    # Instanciar los 3 modelos con sus configuraciones exactas
    modelo_base = UNet(n_channels=1, n_classes=1)
    modelo_hiperbolico = HyperbolicUNet(n_channels=1, n_classes=1, embed_dim=3, c=0.1)
    modelo_krein = KreinUNet(n_channels=1, n_classes=1, dim_pos=1024, dim_neg=512)

    # Definir la lista de evaluación
    modelos_a_evaluar = [
        ("U-Net Baseline (Euclidiana)", modelo_base, "unet_baseline_best.pth"),
        ("U-Net Hiperbólica", modelo_hiperbolico, "unet_hyperbolic_best.pth"),
        ("U-Net Espacio de Kreïn", modelo_krein, "unet_krein_best.pth")
    ]

    # Ejecutar la evaluación secuencialmente
    for nombre, modelo, ruta in modelos_a_evaluar:
        evaluar_modelo(nombre, modelo, ruta, test_loader)

if __name__ == "__main__":
    main()