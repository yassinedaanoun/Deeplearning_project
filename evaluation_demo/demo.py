"""
demo.py — Interface utilisateur pour la prédiction.

Ce fichier contient une interface utilisateur simple basée sur Gradio.
L'utilisateur peut charger une image de radiographie et obtenir une prédiction
(NORMAL ou PNEUMONIA) avec un score de confiance.

Fonctions principales :
- predict : Effectue une prédiction sur une image donnée.
- main : Lance l'interface utilisateur.
"""

import torch
from torchvision import transforms
from PIL import Image
import gradio as gr

def predict(model, image):
    """
    Effectue une prédiction sur une image donnée.

    Args:
        model (torch.nn.Module): Le modèle entraîné.
        image (PIL.Image): L'image de radiographie.

    Returns:
        tuple: (classe prédite, score de confiance)
    """
    # Transformations identiques à celles utilisées pendant l'entraînement
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)  # Ajouter une dimension batch
    model.eval()
    with torch.no_grad():
        output = model(image)
        probs = torch.softmax(output, dim=1)
        confidence, pred_class = torch.max(probs, dim=1)
    return "PNEUMONIA" if pred_class.item() == 1 else "NORMAL", confidence.item()

def main():
    """
    Lance l'interface utilisateur Gradio.
    """
    # Charger le modèle entraîné
    model = torch.load("path_to_trained_model.pth", map_location=torch.device('cpu'))

    # Interface utilisateur avec Gradio
    interface = gr.Interface(
        fn=lambda img: predict(model, Image.open(img)),
        inputs=gr.Image(type="filepath"),
        outputs=["text", "number"],
        title="Chest X-ray Pneumonia Detector",
        description="Chargez une image de radiographie pour obtenir une prédiction (NORMAL ou PNEUMONIA) avec un score de confiance."
    )
    interface.launch()

if __name__ == "__main__":
    main()