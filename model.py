import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """
    Bloque fundamental de la U-Net: (Conv2d -> BatchNorm -> ReLU) repetido 2 veces.
    Mantiene las dimensiones espaciales intactas gracias a padding=1.
    """
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

class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1):
        """
        n_channels: 1 para imágenes de ultrasonido en escala de grises.
        n_classes: 1 para segmentación binaria (Nervio vs. Fondo).
        """
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # --- Codificador (Ruta de contracción) ---
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        
        # --- Cuello de botella ---
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))

        # --- Decodificador (Ruta de expansión) ---
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(1024, 512)
        
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(512, 256)
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(256, 128)
        
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(128, 64)

        # --- Capa de Salida Euclidiana ---
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # 1. Bajada (Encoder)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # 2. Subida (Decoder) con Skip Connections
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
        
        # Características latentes finales justo antes de la clasificación
        features = self.conv4(x) 

        # 3. Predicción
        logits = self.outc(features)
        return logits

# --- Prueba rápida del modelo ---
if __name__ == '__main__':
    # Instanciamos el modelo
    model = UNet(n_channels=1, n_classes=1)
    
    # Creamos un tensor aleatorio simulando un lote del DataLoader que hicimos antes: 
    # Batch=8, Canales=1, Alto=256, Ancho=256
    dummy_input = torch.randn(8, 1, 256, 256)
    
    # Pasamos el tensor por el modelo
    output = model(dummy_input)
    
    print(f"Formato del tensor de entrada: {dummy_input.shape}")
    print(f"Formato del tensor de salida: {output.shape}")