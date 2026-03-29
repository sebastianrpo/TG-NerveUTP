import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model import DoubleConv  # U-net Clásica

# --- 1. Subespacio Positivo (Kernel RBF) ---
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RBFRFF(nn.Module):
    def __init__(self, in_features, out_features, gamma=1.0):
        super().__init__()
        self.out_features = out_features
        
        # 1. Gamma ahora es el único parámetro entrenable
        self.gamma = nn.Parameter(torch.tensor(float(gamma)))
        
        # 2. W_base se registra como buffer (fijo, no entrenable)
        # Se inicializa con una distribución normal estándar
        self.register_buffer('W_base', torch.randn(out_features, in_features))
        
        # 3. b se registra como buffer (fijo, no entrenable)
        # Se inicializa con una distribución uniforme [0, 2π]
        self.register_buffer('b', torch.rand(out_features) * 2 * np.pi)

    def forward(self, x):
        # 4. Escalamos W_base por el gamma entrenable dinámicamente
        W_effective = self.W_base * self.gamma
        proj = F.linear(x, W_effective) + self.b
        return torch.cos(proj) * np.sqrt(2.0 / self.out_features)

# --- 2. Subespacio Negativo (Kernel Polinomial) ---
class PolynomialRFF(nn.Module):
    def __init__(self, in_features, out_features, degree=2):
        super().__init__()
        self.out_features = out_features
        self.degree = degree
        
        # CORRECCIÓN: W debe ser fijo para garantizar la aproximación del kernel.
        # Se registra como buffer para que no se actualice con backpropagation.
        self.register_buffer('W', torch.randn(out_features, in_features))

    def forward(self, x):
        proj = F.linear(x, self.W)
        return (proj ** self.degree) / np.sqrt(self.out_features)

# --- 3. Clasificador en Espacio de Kreïn ---
class KreinClassifier(nn.Module):
    def __init__(self, in_features, n_classes, dim_pos=128, dim_neg=128):
        super().__init__()
        self.rff_pos = RBFRFF(in_features, dim_pos, gamma=0.5)
        self.rff_neg = PolynomialRFF(in_features, dim_neg, degree=2)
        
        # Pesos del hiperplano de clasificación en espacio de Kreïn
        self.w_pos = nn.Parameter(torch.randn(n_classes, dim_pos, 1, 1) * 0.01)
        self.w_neg = nn.Parameter(torch.randn(n_classes, dim_neg, 1, 1) * 0.01)
        self.bias = nn.Parameter(torch.zeros(n_classes))

    def forward(self, x):
        # x: [Batch, Channels, H, W]
        B, C, H, W = x.shape
        x_perm = x.permute(0, 2, 3, 1) # Mover canales al final para aplicar lineal
        
        phi_pos = self.rff_pos(x_perm).permute(0, 3, 1, 2) # [B, dim_pos, H, W]
        phi_neg = self.rff_neg(x_perm).permute(0, 3, 1, 2) # [B, dim_neg, H, W]

        # Producto interno indefinido: <w, phi>_K = w_pos * phi_pos - w_neg * phi_neg
        logit_pos = F.conv2d(phi_pos, self.w_pos)
        logit_neg = F.conv2d(phi_neg, self.w_neg)
        
        # LA MAGIA DEL ESPACIO INDEFINIDO:
        logits = logit_pos - logit_neg + self.bias.view(1, -1, 1, 1)
        
        return logits, self.w_pos, self.w_neg

# --- 4. U-Net de Kreïn Completa ---
class KreinUNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, dim_pos=128, dim_neg=64):
        super(KreinUNet, self).__init__()
        # Encoder
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))

        # Decoder
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(128, 64)

        # Clasificador Kreïn
        self.krein_out = KreinClassifier(in_features=64, n_classes=n_classes, dim_pos=dim_pos, dim_neg=dim_neg)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5)
        x = torch.cat([x4, x], dim=1)
        x = self.conv1(x)
        
        x = self.up2(x)
        x = torch.cat([x3, x], dim=1)
        x = self.conv2(x)
        
        x = self.up3(x)
        x = torch.cat([x2, x], dim=1)
        x = self.conv3(x)
        
        x = self.up4(x)
        x = torch.cat([x1, x], dim=1)
        features = self.conv4(x) 
        
        logits, w_pos, w_neg = self.krein_out(features)
        
        # Durante el entrenamiento retornamos los pesos para la función de pérdida
        if self.training:
            return logits, w_pos, w_neg
        return logits