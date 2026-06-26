"""
Spec2Prop-Edge: Streamlit App
==============================
Lightweight web UI for interactive spectral inference.

Usage:
    streamlit run deploy/app_streamlit.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import numpy as np
import torch

try:
    import streamlit as st
    import matplotlib.pyplot as plt
except ImportError:
    raise ImportError(
        "Streamlit is required: pip install streamlit matplotlib"
    )

from deploy.preprocess_runtime import preprocess_raman_file, preprocess_xrd_file
from deploy.infer_single_sample import load_model, run_inference_torchscript, decode_predictions
from deploy.generate_prediction_report import DISCLAIMER


def main():
    st.set_page_config(page_title="Spec2Prop-Edge", page_icon="🔬", layout="wide")

    st.title("🔬 Spec2Prop-Edge")
    st.markdown("**Edge-Oriented Spectral Inference for Inorganic Materials**")
    st.caption(DISCLAIMER)

    # Sidebar
    st.sidebar.header("Configuration")
    task = st.sidebar.selectbox("Task", ["family", "property", "multimodal"])
    model_file = st.sidebar.text_input("Model file path", "deploy/exported/family_litespecnet_torchscript.pt")
    encoder_file = st.sidebar.text_input("Label encoder path", "deploy/exported/label_encoders.json")
    top_k = st.sidebar.slider("Top-K predictions", 1, 10, 3)
    show_saliency = st.sidebar.checkbox("Show saliency map", value=False)

    # File upload
    col1, col2 = st.columns(2)
    with col1:
        raman_file = st.file_uploader("Upload Raman CSV/TXT", type=["csv", "txt"])
    with col2:
        xrd_file = st.file_uploader("Upload XRD CSV/TXT (optional)", type=["csv", "txt"])

    if raman_file is not None and st.button("🚀 Run Inference", type="primary"):
        # Save uploaded file temporarily
        tmp_raman = os.path.join("deploy", "outputs", "_tmp_raman.csv")
        os.makedirs(os.path.dirname(tmp_raman), exist_ok=True)
        with open(tmp_raman, "wb") as f:
            f.write(raman_file.getvalue())

        try:
            # Preprocess
            with st.spinner("Preprocessing spectrum..."):
                raman_vector, raman_meta = preprocess_raman_file(tmp_raman)

            # Show spectra plots
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            # Raw
            import pandas as pd
            raw_df = pd.read_csv(tmp_raman, comment="#")
            if raw_df.shape[1] >= 2:
                axes[0].plot(raw_df.iloc[:, 0], raw_df.iloc[:, 1], color="#0f3460", linewidth=0.8)
                axes[0].set_title("Raw Spectrum")
                axes[0].set_xlabel("Wavenumber / 2θ")
                axes[0].set_ylabel("Intensity")
            # Preprocessed
            wn = np.linspace(100, 4000, 2048)
            axes[1].plot(wn, raman_vector, color="#e94560", linewidth=0.8)
            axes[1].set_title("Preprocessed (2048 points)")
            axes[1].set_xlabel("Wavenumber (cm⁻¹)")
            axes[1].set_ylabel("Normalized Intensity")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # Load model and run inference
            if not os.path.isfile(model_file):
                st.error(f"Model file not found: {model_file}")
                return

            with st.spinner("Running inference..."):
                model = load_model(model_file, "torchscript")
                raman_tensor = torch.from_numpy(raman_vector).float().unsqueeze(0).unsqueeze(0)

                t0 = time.perf_counter()
                results = run_inference_torchscript(model, raman_tensor)
                t1 = time.perf_counter()
                inference_ms = (t1 - t0) * 1000

            # Decode
            with open(encoder_file) as f:
                all_encoders = json.load(f)
            if "chemical_family_model" in all_encoders:
                encoder = all_encoders["chemical_family_model"]
            else:
                encoder = all_encoders

            probs = list(results.values())[0]
            top_preds = decode_predictions(probs, encoder, top_k)

            # Show results
            st.subheader("Predictions")
            col_a, col_b = st.columns([2, 1])
            with col_a:
                # Bar chart
                classes = [p["class"] for p in top_preds]
                confs = [p["confidence"] for p in top_preds]
                fig2, ax2 = plt.subplots(figsize=(8, 3))
                bars = ax2.barh(classes[::-1], confs[::-1], color="#e94560", alpha=0.8)
                ax2.set_xlim(0, 1)
                ax2.set_xlabel("Confidence")
                ax2.set_title("Top Predictions")
                plt.tight_layout()
                st.pyplot(fig2)
                plt.close()

            with col_b:
                st.metric("Top Prediction", top_preds[0]["class"])
                st.metric("Confidence", f"{top_preds[0]['confidence']:.4f}")
                st.metric("Inference Time", f"{inference_ms:.2f} ms")

            # Saliency
            if show_saliency:
                try:
                    from deploy.saliency_runtime import compute_gradient_saliency
                    saliency = compute_gradient_saliency(model, raman_tensor)
                    sal_norm = saliency / (saliency.max() + 1e-8)
                    fig3, ax3 = plt.subplots(figsize=(12, 3))
                    ax3.fill_between(wn, 0, sal_norm, color="#e94560", alpha=0.5)
                    ax3.plot(wn, raman_vector, color="#1a1a2e", linewidth=0.6)
                    ax3.set_xlabel("Wavenumber (cm⁻¹)")
                    ax3.set_title("Model-Attributed Spectral Regions (Saliency)")
                    plt.tight_layout()
                    st.pyplot(fig3)
                    plt.close()
                except Exception as e:
                    st.warning(f"Saliency failed: {e}")

            # Download report
            report = {
                "sample_id": raman_file.name,
                "task": task,
                "model_name": os.path.basename(model_file),
                "top_predictions": top_preds,
                "inference_time_ms": round(inference_ms, 3),
                "device": "cpu",
                "disclaimer": DISCLAIMER,
            }
            st.download_button(
                "📥 Download Report (JSON)",
                json.dumps(report, indent=2),
                file_name="prediction_report.json",
                mime="application/json"
            )

        except Exception as e:
            st.error(f"Error: {e}")

        finally:
            if os.path.isfile(tmp_raman):
                os.remove(tmp_raman)


if __name__ == "__main__":
    main()
