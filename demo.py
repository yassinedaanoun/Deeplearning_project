"""Local Gradio demo for the chest X-ray pneumonia classifier."""

import argparse

import gradio as gr
import torch
from PIL import Image

from model import load_trained_model
from preprocessing import get_transforms
from train import get_default_device

CLASS_NAMES = ("NORMAL", "PNEUMONIA")
DISCLAIMER = "Prototype d’aide au tri, pas un outil de diagnostic médical."


def _model_device(model: torch.nn.Module, device: torch.device | str | None):
    if device is not None:
        return torch.device(device)
    parameter = next(model.parameters(), None)
    return parameter.device if parameter is not None else torch.device("cpu")


def predict(
    model: torch.nn.Module,
    image: Image.Image,
    device: torch.device | str | None = None,
) -> tuple[str, float]:
    """Return the predicted class name and its softmax score in [0, 1]."""
    if image is None:
        raise ValueError("Upload an X-ray image before requesting a prediction.")
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL image.")

    device = _model_device(model, device)
    model.to(device).eval()
    image_tensor = get_transforms("test")(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        if logits.ndim != 2 or logits.shape != (1, 2):
            raise ValueError("The model must return logits with shape [1, 2].")
        probabilities = torch.softmax(logits, dim=1)[0]
        predicted_index = int(probabilities.argmax().item())

    return CLASS_NAMES[predicted_index], float(probabilities[predicted_index].item())


def build_demo(
    model: torch.nn.Module, device: torch.device | str | None = None
) -> gr.Interface:
    """Build the upload UI without starting a server."""

    def predict_for_ui(image):
        if image is None:
            return "Chargez une radiographie pour obtenir une prédiction.", ""
        label, confidence = predict(model, image, device=device)
        return label, f"{confidence:.1%}"

    return gr.Interface(
        fn=predict_for_ui,
        inputs=gr.Image(type="pil", image_mode="L", sources="upload", label="Radiographie"),
        outputs=[
            gr.Textbox(label="Classe prédite"),
            gr.Textbox(label="Score du modèle"),
        ],
        title="Détecteur de pneumonie sur radiographie thoracique",
        description=DISCLAIMER,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the local chest X-ray demo.")
    parser.add_argument("--checkpoint", default="models/best_model.pth")
    args = parser.parse_args()

    device = get_default_device()
    model = load_trained_model(args.checkpoint, device)
    demo = build_demo(model, device=device)
    demo.launch(server_name="127.0.0.1", share=False)


if __name__ == "__main__":
    main()
