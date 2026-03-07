import torch

def calculate_metrics(preds, targets, threshold=0.5, epsilon=1e-7):
    """
    Calcula las métricas clínicas de segmentación binaria.
    
    Args:
        preds: Tensores de salida del modelo (logits).
        targets: Tensores de la máscara real (Ground Truth).
        threshold: Umbral para convertir probabilidades en valores binarios.
        epsilon: Valor pequeño para evitar divisiones por cero.
    """
    # 1. Aplicar función sigmoide para pasar de logits a probabilidades (0 a 1)
    preds = torch.sigmoid(preds)
    
    # 2. Binarizar usando el umbral
    preds = (preds > threshold).float()
    
    # 3. Aplanar los tensores para operar a nivel de píxel
    preds = preds.view(-1)
    targets = targets.view(-1)
    
    # 4. Calcular la matriz de confusión básica
    TP = (preds * targets).sum()                            # Verdaderos Positivos
    FP = ((preds == 1) & (targets == 0)).sum()              # Falsos Positivos
    TN = ((preds == 0) & (targets == 0)).sum()              # Verdaderos Negativos
    FN = ((preds == 0) & (targets == 1)).sum()              # Falsos Negativos
    
    # 5. Ecuaciones de las métricas
    dice = (2.0 * TP + epsilon) / (2.0 * TP + FP + FN + epsilon)
    iou = (TP + epsilon) / (TP + FP + FN + epsilon)
    sensitivity = (TP + epsilon) / (TP + FN + epsilon)
    specificity = (TN + epsilon) / (TN + FP + epsilon)
    
    # Retornamos los valores como flotantes estándar de Python
    return dice.item(), iou.item(), sensitivity.item(), specificity.item()

def save_checkpoint(state, filename="checkpoint.pth"):
    """
    Guarda el estado actual del modelo y del optimizador.
    Útil para no perder progreso si se interrumpe el entrenamiento.
    """
    print(f"=> Guardando estado del modelo en {filename}")
    torch.save(state, filename)

def load_checkpoint(checkpoint_path, model, optimizer=None):
    """
    Carga los pesos guardados para retomar entrenamiento o hacer inferencia.
    """
    print(f"=> Cargando pesos desde {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer