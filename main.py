from data import get_dataloaders
from model.model import build_model
from train import train_model

# Charger les données
train_loader, val_loader, test_loader = get_dataloaders(batch_size=32)

# Construire et entraîner le modèle
model = build_model(num_classes=2)
trained_model, history = train_model(model, train_loader, val_loader, epochs=10)

# Sauvegarder le modèle entraîné
torch.save(trained_model, "trained_model.pth")