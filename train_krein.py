import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from krein_model import KreinUNet
from dataset import get_dataloaders
from utils import calculate_metrics, save_checkpoint

# --- Hiperparámetros ---
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
NUM_EPOCHS = 50

# Cambiar esta ruta cuando lo subo a Google Colab
#DATA_DIR = '/content/drive/MyDrive/TG_NerveUTP/NerveUTP'
#DATA_DIR = r'C:\Users\srestrepo01\Documents\TG\NerveUTP'
DATA_DIR = r'C:\Users\Sebas\Documents\Dataset'

# --- Función de Pérdida Híbrida Pseudo-Euclidiana ---
class KreinDiceBCELoss(nn.Module):
    def __init__(self, krein_lambda=0.05):
        """
        krein_lambda: Controla qué tanto penalizamos a la red si la 
        energía negativa (Polinomial) supera a la positiva (RBF).
        """
        super().__init__()
        self.krein_lambda = krein_lambda

    def forward(self, logits, targets, w_pos=None, w_neg=None):
        # 1. Pérdida de la Tarea (BCE + Dice estándar)
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        
        bce = F.binary_cross_entropy(probs_flat, targets_flat)
        
        intersection = (probs_flat * targets_flat).sum()
        dice = 1.0 - (2. * intersection + 1e-7) / (probs_flat.sum() + targets_flat.sum() + 1e-7)
        task_loss = bce + dice
        
        # 2. Penalización Topológica (solo se aplica en la fase de entrenamiento)
        if w_pos is not None and w_neg is not None:
            norm_pos = torch.norm(w_pos, p=2)
            norm_neg = torch.norm(w_neg, p=2)
            # Solo suma pérdida si el subespacio negativo se vuelve más grande que el positivo
            krein_penalty = torch.relu(norm_neg - norm_pos)
            return task_loss + self.krein_lambda * krein_penalty
            
        return task_loss

def train_fn(loader, model, optimizer, loss_fn):
    model.train()
    loop = tqdm(loader, desc="Entrenando Kreïn")
    epoch_loss = 0

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().to(device=DEVICE)

        # En modo train, KreinUNet retorna 3 valores
        logits, w_pos, w_neg = model(data)
        
        loss = loss_fn(logits, targets, w_pos, w_neg)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        loop.set_postfix(loss=loss.item())
        
    return epoch_loss / len(loader)

def val_fn(loader, model, loss_fn):
    model.eval()
    val_loss = 0
    total_dice, total_iou, total_sens, total_spec = 0, 0, 0, 0
    loop = tqdm(loader, desc="Validando ")

    with torch.no_grad():
        for data, targets in loop:
            data = data.to(device=DEVICE)
            targets = targets.float().to(device=DEVICE)

            # En modo eval, KreinUNet solo retorna los logits para inferencia rápida
            logits = model(data)
            
            # Evaluamos solo la efectividad de la tarea
            loss = loss_fn(logits, targets)
            val_loss += loss.item()

            dice, iou, sens, spec = calculate_metrics(logits, targets)
            total_dice += dice
            total_iou += iou
            total_sens += sens
            total_spec += spec

    n_batches = len(loader)
    return (val_loss / n_batches, total_dice / n_batches, total_iou / n_batches, 
            total_sens / n_batches, total_spec / n_batches)

def main():
    print(f"=> Iniciando entrenamiento de Modelo de Kreïn en: {DEVICE.upper()}")
    
    transform = A.Compose([
        A.Resize(height=256, width=256),
        ToTensorV2()
    ])

    train_loader, val_loader, test_loader = get_dataloaders(DATA_DIR, transform, BATCH_SIZE)

    # Inicializamos la red con 128 dimensiones para el kernel RBF y 64 para el polinomial
    model = KreinUNet(n_channels=1, n_classes=1, dim_pos=128, dim_neg=64).to(DEVICE)
    loss_fn = KreinDiceBCELoss(krein_lambda=0.05)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_dice = 0.0

    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Época {epoch+1}/{NUM_EPOCHS} ---")
        
        train_loss = train_fn(train_loader, model, optimizer, loss_fn)
        val_loss, val_dice, val_iou, val_sens, val_spec = val_fn(val_loader, model, loss_fn)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Val Dice: {val_dice:.4f} | Val IoU: {val_iou:.4f} | Sens: {val_sens:.4f} | Spec: {val_spec:.4f}")

        if val_dice > best_dice:
            best_dice = val_dice
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            # Guardamos el modelo con un nombre específico para esta arquitectura
            save_checkpoint(checkpoint, filename="unet_krein_best.pth")

if __name__ == "__main__":
    main()