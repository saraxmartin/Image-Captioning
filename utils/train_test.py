import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def initiliaze_storage():
    pass


def train_model(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    correct, total = 0, 0

    for images, captions in train_loader:
        images, captions = images.to(DEVICE), captions.to(DEVICE)
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

    pass


def test_model():
    pass

