"""
Spec2Prop-Edge: Gradient Saliency Visualization
=================================================
Compute and plot gradient-based saliency maps for Raman-only CNN models.

Wording note: These are "model-attributed spectral regions", not
"confirmed physical peaks" — saliency shows what the model uses,
not necessarily what is physically meaningful.
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def compute_gradient_saliency(model, raman_tensor: torch.Tensor) -> np.ndarray:
    """
    Compute gradient-based saliency for a 1D spectral input.

    Parameters
    ----------
    model : torch.nn.Module
        The CNN model (must accept input of shape [1, 1, N]).
    raman_tensor : torch.Tensor, shape (1, 1, N)
        Preprocessed Raman spectrum tensor.

    Returns
    -------
    saliency : np.ndarray, shape (N,)
        Absolute gradient saliency values.
    """
    model.eval()
    inp = raman_tensor.clone().detach().requires_grad_(True)

    output = model(inp)
    if isinstance(output, dict):
        # Multi-task head: use the first head
        key = list(output.keys())[0]
        logits = output[key]
    else:
        logits = output

    # Get the predicted class score
    pred_class = logits.argmax(dim=1)
    score = logits[0, pred_class[0]]

    model.zero_grad()
    score.backward()

    saliency = inp.grad.data.abs().squeeze().cpu().numpy()
    return saliency


def plot_saliency(
    spectrum: np.ndarray,
    saliency: np.ndarray,
    output_path: str,
    wn_min: float = 100,
    wn_max: float = 4000,
    title: str = "Raman Spectrum with Model-Attributed Spectral Regions",
):
    """
    Plot the Raman spectrum with saliency overlay.

    Parameters
    ----------
    spectrum : np.ndarray, shape (N,)
        Preprocessed spectrum.
    saliency : np.ndarray, shape (N,)
        Saliency values.
    output_path : str
        Path to save the plot.
    """
    n = len(spectrum)
    wavenumbers = np.linspace(wn_min, wn_max, n)

    # Normalize saliency for visualization
    sal_norm = saliency / (saliency.max() + 1e-8)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})

    # Top: spectrum with saliency fill
    ax1.plot(wavenumbers, spectrum, color="#1a1a2e", linewidth=0.8, label="Preprocessed Spectrum")
    ax1.fill_between(wavenumbers, 0, spectrum, alpha=0.15, color="#0f3460")

    # Overlay saliency as color
    for i in range(n - 1):
        alpha = float(sal_norm[i]) * 0.7
        ax1.axvspan(wavenumbers[i], wavenumbers[i + 1], alpha=alpha, color="#e94560", zorder=0)

    ax1.set_ylabel("Normalized Intensity")
    ax1.set_title(title, fontsize=11, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=8)

    # Bottom: saliency magnitude
    ax2.fill_between(wavenumbers, 0, sal_norm, color="#e94560", alpha=0.6)
    ax2.plot(wavenumbers, sal_norm, color="#e94560", linewidth=0.5)
    ax2.set_xlabel("Wavenumber (cm⁻¹)")
    ax2.set_ylabel("Saliency")
    ax2.set_ylim(0, 1.05)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saliency plot saved: {output_path}")

    return output_path
