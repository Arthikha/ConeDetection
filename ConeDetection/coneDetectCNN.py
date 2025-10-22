import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np

from sklearn.metrics import average_precision_score

class ConeDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.image_dir = os.path.join(root, 'fsoco_bounding_boxes_train_yolo-gen', 'images')
        self.label_dir = os.path.join(root, 'fsoco_bounding_boxes_train_yolo-gen', 'labels')
        self.image_files = [f for f in os.listdir(self.image_dir) if f.endswith('.jpg') or f.endswith('.png')]
        with open(os.path.join(root, 'classes.txt'), 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        label_path = os.path.join(self.label_dir, img_name.replace('.jpg', '.txt').replace('.png', '.txt'))
        
        img = Image.open(img_path).convert('RGB')
        width, height = img.size

        # Parsing YOLO annotations
        boxes = []
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    class_id, x_center, y_center, width_box, height_box = map(float, line.split())
                    # Converting YOLO normalized coordinates to absolute coordinates [x_min, y_min, x_max, y_max]
                    x_min = (x_center - width_box / 2) * width
                    y_min = (y_center - height_box / 2) * height
                    x_max = (x_center + width_box / 2) * width
                    y_max = (y_center + height_box / 2) * height
                    boxes.append([x_min, y_min, x_max, y_max])
                    labels.append(int(class_id))

        # Padding to 10 boxes
        max_boxes = 10
        while len(boxes) < max_boxes:
            boxes.append([0, 0, 0, 0])
            labels.append(-1)  # Invalid class for padding
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        if self.transform:
            img = self.transform(img)

        target = {'boxes': boxes, 'labels': labels}
        return img, target

# CNN
class CNN(nn.Module):
    def __init__(self, num_classes=2, max_boxes=10):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),  # RGB
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64 * 80 * 80, 256)  
        self.box_head = nn.Linear(256, max_boxes * 4)  # Predicting x_min, y_min, x_max, y_max for each box
        self.cls_head = nn.Linear(256, max_boxes * num_classes)  # Predicting class scores

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = torch.relu(self.fc(x))
        boxes = self.box_head(x).view(-1, 10, 4)
        scores = self.cls_head(x).view(-1, 10, num_classes)
        return boxes, scores

# Loss Function
def compute_loss(pred_boxes, pred_scores, target_boxes, target_labels):
    box_loss = nn.SmoothL1Loss()(pred_boxes, target_boxes)  # Bounding box regression
    cls_loss = nn.CrossEntropyLoss(ignore_index=-1)(pred_scores.view(-1, num_classes), target_labels.view(-1))
    return box_loss + cls_loss

# Training
transform = transforms.Compose([transforms.Resize((640, 640)), transforms.ToTensor()])
dataset = ConeDataset('/kaggle/input/formula-student-cones-detection-irt', transform=transform)
data_loader = DataLoader(dataset, batch_size=2, shuffle=True)

# Setting num_classes based on classes.txt
num_classes = len(dataset.classes)
model = CNN(num_classes=num_classes, max_boxes=10)
optimizer = optim.Adam(model.parameters(), lr=0.001)
device = torch.device('cpu')
model.to(device)

# Training Loop
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for images, targets in data_loader:
        images = images.to(device)
        target_boxes = targets['boxes'].to(device)
        target_labels = targets['labels'].to(device)

        optimizer.zero_grad()
        pred_boxes, pred_scores = model(images)
        loss = compute_loss(pred_boxes, pred_scores, target_boxes, target_labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(data_loader)}")

# Saving Model
torch.save(model.state_dict(), 'cone_detector.pth')


# Evaluation
model.eval()
ap_scores = []
with torch.no_grad():
    for images, targets in data_loader:
        images = images.to(device)
        pred_boxes, pred_scores = model(images)
        pred_labels = torch.argmax(pred_scores, dim=2).cpu().numpy()
        # Confidence for positive class
        pred_conf = torch.softmax(pred_scores, dim=2)[:, :, 1].cpu().numpy()  
        true_labels = targets['labels'].cpu().numpy()
        for i in range(len(pred_labels)):
            ap = average_precision_score((true_labels[i] >= 0).astype(int), pred_conf[i])
            ap_scores.append(ap)
print(f"mAP@0.5: {np.mean(ap_scores):.4f}")


# Inference
model.eval()
img = Image.open('/kaggle/input/formula-student-cones-detection-irt/fsoco_bounding_boxes_train_yolo-gen/images/test_image.jpg').convert('RGB')
img = transform(img).unsqueeze(0).to(device)
with torch.no_grad():
    boxes, scores = model(img)
    pred_labels = torch.argmax(scores, dim=2)
    pred_conf = torch.softmax(scores, dim=2)[:, :, 1]
for i, box in enumerate(boxes[0]):
    if pred_conf[0][i] > 0.5 and pred_labels[0][i] != -1:
        print(f"Cone: {dataset.classes[pred_labels[0][i]]}, Box: {box.tolist()}, Conf: {pred_conf[0][i]:.2f}")
