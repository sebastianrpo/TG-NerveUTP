import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. Funciones matemáticas hiperbólicas del artículo
def expmap0(x, c):
    """
    Mapa exponencial: Proyecta vectores del espacio euclidiano a la Bola de Poincaré.
    (Ecuación 4 del artículo).
    """
    c_tensor = torch.tensor(c, device=x.device, dtype=x.dtype)
    sqrt_c = torch.sqrt(c_tensor)
    norm_x = torch.norm(x, p=2, dim=1, keepdim=True)
    norm_x = torch.clamp(norm_x, min=1e-5) # Evitar división por cero
    
    return torch.tanh(sqrt_c * norm_x) * (x / (sqrt_c * norm_x))

class TractableHyperbolicClassifier(nn.Module):
    """
    Clasificador a nivel de píxel usando regresión logística multinomial hiperbólica.
    (Implementa la versión tratable en memoria del artículo.)
    """
    def __init__(self, in_features, num_classes, c=0.1):
        super().__init__()
        self.c = c
        self.num_classes = num_classes
        
        # Parámetros del giroplano: p (desplazamiento) y w (orientación)
        self.p = nn.Parameter(torch.randn(num_classes, in_features) * 1e-3)
        self.w = nn.Parameter(torch.randn(num_classes, in_features) * 1e-3)

    def forward(self, x):
        # x shape: [Batch, Dims, Alto, Ancho]
        B, D, H, W = x.shape
        c = self.c

        # 1. Proyectar a la bola de Poincaré
        z = expmap0(x, c)
        z_norm2 = torch.norm(z, p=2, dim=1, keepdim=True).pow(2)

        logits = []
        for y in range(self.num_classes):
            p_y = self.p[y] 
            w_y = self.w[y] 
            
            # \hat{p}_y = -p_y (como se define en la Sec 3.2)
            p_hat = -p_y.view(1, D, 1, 1)
            w_y_view = w_y.view(1, D, 1, 1)

            p_hat_norm2 = torch.norm(p_hat, p=2).pow(2)
            w_y_norm = torch.norm(w_y, p=2)

            inner_pz = (p_hat * z).sum(dim=1, keepdim=True)

            # Ecuación 9: Factores alpha y beta para evitar la suma de Möbius directa
            num_alpha = 1 + 2 * c * inner_pz + c * z_norm2
            denom = 1 + 2 * c * inner_pz + (c**2) * p_hat_norm2 * z_norm2
            denom = torch.clamp(denom, min=1e-5)

            alpha = num_alpha / denom
            beta = (1 - c * p_hat_norm2) / denom

            # Ecuación 10: Producto interno reescrito
            inner_pw = (p_hat * w_y_view).sum()
            inner_zw = (z * w_y_view).sum(dim=1, keepdim=True)
            inner_mobius_w = alpha * inner_pw + beta * inner_zw

            # Ecuación 11: Norma cuadrada de la adición de Möbius
            mobius_norm2 = (alpha**2) * p_hat_norm2 + 2 * alpha * beta * inner_pz + (beta**2) * z_norm2

            # Ecuación 7: Cálculo del logit (distancia al giroplano)
            lambda_p = 2 / (1 - c * p_hat_norm2)
            
            # Argumento del arcoseno hiperbólico
            arg_num = 2 * torch.sqrt(torch.tensor(c, device=x.device)) * inner_mobius_w
            arg_den = (1 - c * mobius_norm2) * w_y_norm + 1e-5
            arg = torch.clamp(arg_num / arg_den, -1e4, 1e4) # Estabilidad numérica

            zeta = (lambda_p * w_y_norm / torch.sqrt(torch.tensor(c, device=x.device))) * torch.asinh(arg)
            logits.append(zeta)

        return torch.cat(logits, dim=1)

# 2. Reutilizamos el bloque DoubleConv de la U-Net original
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

# 3. La nueva U-Net Hiperbólica
class HyperbolicUNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, embed_dim=3, c=0.1):
        super(HyperbolicUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        # El artículo sugiere c=0.1 como buen punto de partida [cite: 610]

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

        # Capa de reducción de dimensiones (Ej. de 64 a 3 canales)
        self.to_embed_dim = nn.Conv2d(64, embed_dim, kernel_size=1)
        
        #Clasificador Hiperbólico final
        self.hyperbolic_out = TractableHyperbolicClassifier(in_features=embed_dim, num_classes=n_classes, c=c)

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
        
        # Reducimos las dimensiones y aplicamos clasificación hiperbólica
        embed_features = self.to_embed_dim(features)
        logits = self.hyperbolic_out(embed_features)
        
        return logits