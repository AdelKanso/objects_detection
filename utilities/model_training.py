import os
import cv2
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm

# ----------------------
# Dataset Class
# ----------------------
class FaceDetectionDataset(Dataset):
    def __init__(self, image_folder, label_folder, transform=None):
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        label_file = image_file.replace('.jpg', '.txt')
        image_path = os.path.join(self.image_folder, image_file)
        label_path = os.path.join(self.label_folder, label_file)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image, (224, 224))
        image_resized = to_tensor(image_resized)

        boxes = []
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        label, x, y, box_w, box_h = map(float, parts)
                        if int(label) == 0:  # Class 0 is "face"
                            x_min = (x - box_w / 2) * 224
                            y_min = (y - box_h / 2) * 224
                            x_max = (x + box_w / 2) * 224
                            y_max = (y + box_h / 2) * 224
                            boxes.append([x_min, y_min, x_max, y_max])
                            labels.append(1)  # Face class

        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.empty((0, 4), dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64) if labels else torch.empty((0,), dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels}
        return image_resized, target

# ----------------------
# Simple CNN Backbone
# ----------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.out_channels = 128

    def forward(self, x):
        return self.features(x)

# ----------------------
# Initialize and Train the Model
# ----------------------
if __name__ == "__main__":
    # Paths
    TRAIN_IMAGES_PATH = "C:/Users/lenovo/Downloads/Face_Detection_DataSet/images/val"
    TRAIN_ANNOTS_PATH = "C:/Users/lenovo/Downloads/Face_Detection_DataSet/labels/val"

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset and DataLoader
    dataset = FaceDetectionDataset(TRAIN_IMAGES_PATH, TRAIN_ANNOTS_PATH)
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

    # Initialize the SimpleCNN Backbone
    backbone = SimpleCNN()
    backbone.out_channels = 128  # Ensure the output matches what FasterRCNN expects

    # Construct the model
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))
    roi_pooler = MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)

    model = FasterRCNN(
        backbone=backbone,
        num_classes=2,  # Background + Face
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler
    )
    model.to(device)

    # Optimizer and Learning Rate Scheduler
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Training Loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for images, targets in tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()

        print(f"Epoch {epoch+1} Loss: {epoch_loss/len(data_loader):.4f}")
        lr_scheduler.step()

    # Save the trained model
    torch.save(model.state_dict(), "FaceDetectionModel.pth")
    print("Model saved as 'FaceDetectionModel.pth'")
