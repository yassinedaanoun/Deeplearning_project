"""
evaluate.py — Évaluation des performances du modèle.

Ce fichier contient les fonctions nécessaires pour évaluer un modèle
entraîné sur l'ensemble de test. Les métriques calculées incluent :
- Matrice de confusion
- Précision
- Rappel
- F1-score

Fonctions principales :
- evaluate_model : Évalue le modèle sur un DataLoader de test.
- plot_confusion_matrix : Affiche graphiquement la matrice de confusion.
"""

import torch
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, test_loader):
    """
    Évalue les performances du modèle sur l'ensemble de test.

    Args:
        model (torch.nn.Module): Le modèle entraîné.
        test_loader (torch.utils.data.DataLoader): DataLoader contenant les données de test.

    Returns:
        dict: Un dictionnaire contenant les métriques suivantes :
            - confusion_matrix : Matrice de confusion.
            - precision : Précision du modèle.
            - recall : Rappel du modèle.
            - f1_score : Score F1 du modèle.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calcul des métriques
    cm = confusion_matrix(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    return {
        "confusion_matrix": cm,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

def plot_confusion_matrix(cm, class_names):
    """
    Affiche graphiquement la matrice de confusion.

    Args:
        cm (numpy.ndarray): Matrice de confusion.
        class_names (list): Liste des noms des classes.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Prédictions")
    plt.ylabel("Vérités terrain")
    plt.title("Matrice de confusion")
    plt.show()