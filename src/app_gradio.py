from __future__ import annotations

import gradio as gr
from PIL import Image

from .config import DEFAULT_MODEL_PATH, DEFAULT_CLASS_NAMES_PATH, IMG_SIZE
from .inference import load_model, load_class_names, predict_image


PRIVACY_TEXT = """
### Privacy & Consent
By proceeding, you confirm you have the right to upload this image and consent to its processing **in this session** for emotion prediction.
- The app processes images for inference only.
- Do **not** upload sensitive or identifiable images without permission.
"""


def build_app():
    # Load once at startup (fast inference; no training)
    model = load_model(DEFAULT_MODEL_PATH)
    class_names = load_class_names(DEFAULT_CLASS_NAMES_PATH)

    with gr.Blocks(title="HFE Emotion Classifier (Inference)") as demo:
        gr.Markdown("# Human Face Emotion Classifier — Gradio (Prediction Only)")
        gr.Markdown(PRIVACY_TEXT)

        consent = gr.Checkbox(label="I agree / I have consent to process this image", value=False)

        with gr.Row():
            img_in = gr.Image(type="pil", label="Upload Image", interactive=True)
            label_out = gr.Textbox(label="Predicted Emotion")
        probs_out = gr.Label(num_top_classes=5, label="Class Probabilities")

        def _guarded_predict(consent_ok: bool, img: Image.Image):
            if not consent_ok:
                return gr.update(value="Consent required"), {}
            scores, pred = predict_image(model, class_names, img, IMG_SIZE)
            return pred, scores

        btn = gr.Button("Predict")
        btn.click(
            fn=_guarded_predict,
            inputs=[consent, img_in],
            outputs=[label_out, probs_out],
        )

        gr.Markdown(
            "**Note:** Model + labels are loaded from `models/`. "
            "Make sure `class_names.json` matches the training label order."
        )

    return demo


def main():
    demo = build_app()
    demo.launch()


if __name__ == "__main__":
    main()
