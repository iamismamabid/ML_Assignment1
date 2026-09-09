
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32

train_pipeline = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.RandomRotation(degrees=10),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

val_test_pipeline = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


data_root = '.'
train_set = datasets.ImageFolder(
    os.path.join(data_root, 'train'), transform=train_pipeline
)
val_set = datasets.ImageFolder(
    os.path.join(data_root, 'val'), transform=val_test_pipeline
)
test_set = datasets.ImageFolder(
    os.path.join(data_root, 'test'), transform=val_test_pipeline
)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)



class ChestXRayClassifier(nn.Module):

  def __init__(self):
    super(ChestXRayClassifier, self).__init__()
    self.conv_blocks = nn.Sequential(
        # Block 1
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),  # 64x64
        # Block 2
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),  # 32x32
        # Block 3
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),  # 16x16
    )
    self.fc_head = nn.Sequential(
        nn.Dropout(0.35),
        nn.Linear(128 * 16 * 16, 128),
        nn.ReLU(),
        nn.Dropout(0.25),
        nn.Linear(128, 1),
    )

  def forward(self, x):
    features = self.conv_blocks(x)
    flattened = features.view(features.size(0), -1)
    return self.fc_head(flattened)


model = ChestXRayClassifier().to(device)

try:
  from torchinfo import summary

  print('\n=== Model Architecture Summary ===')
  summary(model, input_size=(1, 3, 128, 128))
except ImportError:
  os.system('pip install -q torchinfo')
  from torchinfo import summary

  print('\n=== Model Architecture Summary ===')
  summary(model, input_size=(1, 3, 128, 128))


loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.5]).to(device))
optimizer = optim.Adam(model.parameters(), lr=0.0008)

EPOCHS = 5
print('\nStarting Model Training...')
for epoch in range(EPOCHS):
  model.train()
  total_loss = 0.0
  for imgs, labels in train_loader:
    imgs, labels = imgs.to(device), labels.float().unsqueeze(1).to(device)
    optimizer.zero_grad()
    preds = model(imgs)
    loss = loss_fn(preds, labels)
    loss.backward()
    optimizer.step()
    total_loss += loss.item() * imgs.size(0)
  print(
      f'Epoch [{epoch+1}/{EPOCHS}] - Average Loss:'
      f' {total_loss/len(train_set):.4f}'
  )


model.eval()
all_labels, all_preds = [], []
with torch.no_grad():
  for imgs, labels in test_loader:
    imgs = imgs.to(device)
    outputs = model(imgs)
    predicted_classes = (torch.sigmoid(outputs) > 0.5).int().cpu().numpy()
    all_labels.extend(labels.numpy())
    all_preds.extend(predicted_classes.flatten())

print('\n--- Test Set Classification Report ---')
print(
    classification_report(
        all_labels, all_preds, target_names=['Normal', 'Pneumonia']
    )
)


cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Predicted Normal', 'Predicted Pneumonia'],
    yticklabels=['Actual Normal', 'Actual Pneumonia'],
)
plt.title('Confusion Matrix: Pneumonia Detection')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
