import torch
from torchvision import transforms

def get_train_transforms():
    """
    Retourne le pipeline de prétraitement pour l'entraînement. 
    Inclut la Data Augmentation pour éviter le surapprentissage.
    """
    return transforms.Compose([
        # 1. Conversion en 3 canaux (requis par ResNet)
        transforms.Grayscale(num_output_channels=3), 
        
        # 2. Data Augmentation
        transforms.RandomResizedCrop(224, scale=(0.85, 1.0)), # Recadrage et mise à 224x224
        transforms.RandomRotation(degrees=15),                # Rotation aléatoire
        transforms.RandomHorizontalFlip(p=0.5),               # Effet miroir aléatoire
        
        # 3. Conversion en tenseur PyTorch
        transforms.ToTensor(),
        
        # 4. Normalisation avec les poids d'ImageNet
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

def get_val_transforms():
    """
    Retourne le pipeline de prétraitement pour la validation et le test. 
    Aucune Data Augmentation ici, juste le redimensionnement et la normalisation.
    """
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)), # Redimensionnement strict
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

def get_loss_weights(num_normal=1583, num_pneumonia=4273):
    """
    Calcule les poids pour la fonction de perte (CrossEntropyLoss) afin de gérer 
    le déséquilibre des classes (3x plus de pneumonies que de cas normaux).
    """
    weight_normal = num_pneumonia / num_normal
    weight_pneumonia = 1.0
    
    return torch.tensor([weight_normal, weight_pneumonia], dtype=torch.float)