import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model import UNet
from dataset import get_dataloaders
from utils import calculate_metrics, save_checkpoint

# --- Hiperparámetros ---
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
NUM_EPOCHS = 50 
#DATA_DIR = r'C:\Users\srestrepo01\Documents\TG\NerveUTP'
DATA_DIR = r'C:\Users\Sebas\Documents\Dataset'

# --- Función de Pérdida Combinada (Crucial para Ultrasonido) ---
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # 1. Pasamos los logits por la sigmoide
        inputs = torch.sigmoid(inputs)
        
        # 2. Aplanamos los tensores
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # 3. BCE (Binary Cross Entropy)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        
        # 4. Dice Loss
        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        
        # Combinamos ambas
        return BCE + dice_loss

def train_fn(loader, model, optimizer, loss_fn):
    model.train()
    loop = tqdm(loader, desc="Entrenando")
    epoch_loss = 0

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().to(device=DEVICE)

        # Forward pass normal (sin scaler de GPU para evitar warnings en CPU)
        predictions = model(data)
        loss = loss_fn(predictions, targets)

        # Backward pass
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

            predictions = model(data)
            loss = loss_fn(predictions, targets)
            val_loss += loss.item()

            dice, iou, sens, spec = calculate_metrics(predictions, targets)
            total_dice += dice
            total_iou += iou
            total_sens += sens
            total_spec += spec

    n_batches = len(loader)
    return (val_loss / n_batches, total_dice / n_batches, total_iou / n_batches, 
            total_sens / n_batches, total_spec / n_batches)

def main():
    print(f"=> Iniciando entrenamiento en: {DEVICE.upper()}")
    
    transform = A.Compose([
        A.Resize(height=256, width=256),
        ToTensorV2()
    ])

    train_loader, val_loader, test_loader = get_dataloaders(DATA_DIR, transform, BATCH_SIZE)

    model = UNet(n_channels=1, n_classes=1).to(DEVICE)
    # Usamos nuestra nueva pérdida combinada
    loss_fn = DiceBCELoss()
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
            save_checkpoint(checkpoint, filename="unet_baseline_best.pth")

if __name__ == "__main__":
    main()