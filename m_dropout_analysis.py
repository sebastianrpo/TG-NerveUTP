import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from krein_model import KreinUNet
from dataset import get_dataloaders

# --- Configuración ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = r'C:\Users\Sebas\Documents\Dataset'
MODEL_PATH = "unet_krein_best.pth"
N_INFERENCES = 150
DROPOUT_RATE = 0.3
SEED = 42


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def inject_dropout(model: nn.Module, p: float = 0.3):
    """
    Inserta Dropout2d UNA VEZ después de cada ReLU en los bloques DoubleConv
    Omite inserción si ya existe Dropout.

    Estructura por DoubleConv:
        Conv2d -> BN -> ReLU -> Dropout2d(p)
        Conv2d -> BN -> ReLU -> Dropout2d(p)
    """
    for _, child in model.named_children():
        if child.__class__.__name__ == 'DoubleConv':
            new_layers = []
            seq = list(child.double_conv)
            for i, m in enumerate(seq):
                new_layers.append(m)
                if isinstance(m, nn.ReLU):
                    next_is_dropout = (
                        i + 1 < len(seq)
                        and isinstance(seq[i + 1], (nn.Dropout2d, nn.Dropout))
                    )
                    if not next_is_dropout:
                        new_layers.append(nn.Dropout2d(p=p))
            child.double_conv = nn.Sequential(*new_layers)
        else:
            inject_dropout(child, p)


def enable_mc_dropout(model: nn.Module):
    """Congela BN pero mantiene Dropout activo para MC Dropout."""
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout2d, nn.Dropout)):
            m.train()


def normalize_01(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-8)


def compute_neg_energy_map(
    features: torch.Tensor,
    krein_layer: nn.Module
) -> np.ndarray:
    """
    Reconstruye el mapa espacial de energía negativa a partir de las
    features de entrada a krein_out.

    El forward interno de KreinLayer es:
        feat_perm = features.permute(0,2,3,1)     [1, H, W, C]
        phi_neg   = rff_neg(feat_perm)             [1, H, W, D_neg]
        phi_neg_t = phi_neg.permute(0,3,1,2)       [1, D_neg, H, W]
        w_neg_map = conv2d(phi_neg_t, w_neg)       [1, 1, H, W]

    w_neg tiene shape [1, D_neg, 1, 1], es un kernel de conv2d 1×1.
    """
    with torch.no_grad():
        feat_perm = features.permute(0, 2, 3, 1)          # [1, H, W, C]
        phi_neg   = krein_layer.rff_neg(feat_perm)         # [1, H, W, D_neg]
        phi_neg_t = phi_neg.permute(0, 3, 1, 2)           # [1, D_neg, H, W]
        w_neg_map = F.conv2d(phi_neg_t, krein_layer.w_neg) # [1, 1, H, W]

    neg_np = w_neg_map.squeeze().cpu().numpy()             # [H, W]
    return normalize_01(neg_np)


def main():
    set_seed(SEED)
    print(f"=> Iniciando Análisis MC Dropout en: {DEVICE.upper()}")

    # ------------------------------------------------------------------
    # 1-2. Cargar modelo y pesos
    # ------------------------------------------------------------------
    model = KreinUNet(n_channels=1, n_classes=1, dim_pos=1024, dim_neg=512).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    print("[*] Pesos originales cargados.")

    # ------------------------------------------------------------------
    # 3. Inyectar Dropout después de cargar pesos
    # ------------------------------------------------------------------
    inject_dropout(model, p=DROPOUT_RATE)
    model = model.to(DEVICE)
    print(f"[*] Dropout2d (p={DROPOUT_RATE}) inyectado después de cada ReLU en DoubleConv.")

    # ------------------------------------------------------------------
    # 4. Hook sobre la ENTRADA de krein_out.
    #
    #
    #    Capturamos input[0] (el tensor 'features' [1, C, H, W]) y luego
    #    llamamos a compute_neg_energy_map() que replica el forward interno.
    # ------------------------------------------------------------------
    krein_inputs: list[torch.Tensor] = []

    def krein_input_hook(module, input, output):
        # input es una tupla; input[0] son las features [1, C, H, W]
        krein_inputs.append(input[0].detach().cpu())

    hook_handle = model.krein_out.register_forward_hook(krein_input_hook)

    # ------------------------------------------------------------------
    # 5. Cargar datos
    # ------------------------------------------------------------------
    transform = A.Compose([A.Resize(height=256, width=256), ToTensorV2()])
    _, _, test_loader = get_dataloaders(DATA_DIR, transform, batch_size=1)

    dataset = test_loader.dataset
    idx = random.randint(0, len(dataset) - 1)
    sample = dataset[idx]

    if len(sample) != 3:
        raise ValueError(
            f"dataset[i] debe devolver (image, mask, name), "
            f"pero devolvió tupla de longitud {len(sample)}."
        )
    image_tensor, mask_tensor, img_name = sample
    print(f"[*] Analizando imagen: {img_name}")

    img_input = image_tensor.unsqueeze(0).to(DEVICE)

    # ==========================================
    # PASO A: Predicción limpia + mapa de energía negativa
    # ==========================================
    model.eval()
    krein_inputs.clear()

    with torch.no_grad():
        clean_logits = model(img_input)
        clean_pred = torch.sigmoid(clean_logits).squeeze().cpu().numpy()

        # Reconstruir mapa espacial de energía negativa [H, W]
        features_in = krein_inputs[0].to(DEVICE)  # [1, C, H, W]
        logit_neg_norm = compute_neg_energy_map(features_in, model.krein_out)
        print(f"[*] Mapa energía negativa shape: {logit_neg_norm.shape}  "
              f"rango: [{logit_neg_norm.min():.3f}, {logit_neg_norm.max():.3f}]")

    print("[*] Predicción limpia y energía negativa extraídas.")

    # ==========================================
    # PASO B: Inferencia Monte Carlo (N=150)
    # ==========================================
    enable_mc_dropout(model)
    mc_predictions: list[torch.Tensor] = []
    krein_inputs.clear()

    print(f"[*] Ejecutando {N_INFERENCES} inferencias con MC Dropout...")
    with torch.no_grad():
        for _ in tqdm(range(N_INFERENCES)):
            logits = model(img_input)
            probs = torch.sigmoid(logits).squeeze().cpu()
            mc_predictions.append(probs)
            krein_inputs.clear()

    hook_handle.remove()

    mc_stacked = torch.stack(mc_predictions)           # [N, H, W]
    mean_map   = torch.mean(mc_stacked, dim=0).numpy()
    std_map    = torch.std(mc_stacked,  dim=0).numpy()

    # ==========================================
    # PASO C: Visualización
    # ==========================================
    img_show = normalize_01(image_tensor.squeeze().cpu().numpy())

    fig, axes = plt.subplots(1, 5, figsize=(30, 6))
    fig.suptitle(f'Análisis MC Dropout — {img_name}', fontsize=13, y=1.01)

    panels = [
        (img_show,       'gray', f'Ultrasonido Original\n({img_name})'),
        (clean_pred,     'jet',  'Predicción Limpia\n(sin Dropout)'),
        (mean_map,       'jet',  f'Predicción Media MC\n(N={N_INFERENCES})'),
        (logit_neg_norm, 'jet',  'Energía Negativa (w_neg · φ)\n(Rama Kreïn, normalizada)'),
        (std_map,        'jet',  'Incertidumbre Epistémica\n(Desv. Std. MC)'),
    ]

    for ax, (data, cmap, title) in zip(axes, panels):
        im = ax.imshow(data, cmap=cmap, vmin=0, vmax=1 if cmap != 'hot' else None)
        ax.set_title(title, fontsize=10)
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------------
    # Reporte de correlación: w_neg ↔ incertidumbre MC
    # ------------------------------------------------------------------
    corr     = np.corrcoef(logit_neg_norm.ravel(), std_map.ravel())[0, 1]
    abs_corr = abs(corr)
    print(f"\n[Resultado] Correlación de Pearson (w_neg vs. std MC): {corr:.4f}")

    direction = (
        "BAJA energía negativa (repulsión Kreïn → zona del nervio)"
        if corr < 0 else "ALTA energía negativa"
    )

    if abs_corr > 0.5:
        print(f"   → Correlación fuerte ({corr:+.4f}): incertidumbre alineada con {direction} ✓")
    elif abs_corr > 0.2:
        print(f"   → Correlación moderada ({corr:+.4f}): relación parcial entre ambas señales.")
    else:
        print(f"   → Correlación débil ({corr:+.4f}): señales no alineadas espacialmente.")


if __name__ == '__main__':
    main()