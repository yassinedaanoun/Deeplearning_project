# Fichier : test_preprocessing.py
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Import depuis ton propre fichier de preprocessing
from preprocessing import get_train_transforms

def imshow_tensor(tensor, title=None):
    """Fonction utilitaire pour annuler la normalisation et afficher l'image"""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    # Conversion Tensor -> Numpy
    image = tensor.numpy().transpose((1, 2, 0))
    image = std * image + mean # Dé-normalisation
    image = np.clip(image, 0, 1) # Maintien entre 0 et 1
    
    plt.imshow(image)
    if title:
        plt.title(title)
    plt.axis('off')

if __name__ == "__main__":
    # 1. Chargement du pipeline d'entraînement
    print("Chargement des transformations...")
    train_transforms = get_train_transforms()

    # 2. Création d'une image "Lambda" (bruit aléatoire de 800x600)
    dummy_pixels = np.random.randint(0, 255, (600, 800), dtype=np.uint8)
    dummy_image = Image.fromarray(dummy_pixels, 'L') # 'L' direct pour éviter le warning Pillow

    # 3. Application du pipeline
    transformed_tensor = train_transforms(dummy_image)

    # 4. Vérification dans le terminal
    print("-" * 30)
    print(f"Format attendu par ResNet18 : torch.Size([3, 224, 224])")
    print(f"Format réel obtenu          : {transformed_tensor.shape}")
    print("-" * 30)

    # 5. Affichage Avant / Après
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
    ax[0].imshow(dummy_image, cmap='gray')
    ax[0].set_title('Image Lambda (Avant)')
    ax[0].axis('off')

    plt.sca(ax[1])
    imshow_tensor(transformed_tensor, 'Tenseur Transformé (Après)')
    
    plt.show()