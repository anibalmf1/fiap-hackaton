import os
import torch
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torch import nn, optim
from ultralytics import YOLO
from app.utils import extract_frames

MODEL_DIR = "models"
DETECT_PATH = os.path.join(MODEL_DIR, "detect.pt")
CLASSIFY_PATH = os.path.join(MODEL_DIR, "classify.pt")

class ModelHandler:
    def __init__(self, device="cpu"):
        os.makedirs(MODEL_DIR, exist_ok=True)

        if os.path.exists(DETECT_PATH):
            self.detector = YOLO(DETECT_PATH)
        else:
            self.detector = YOLO("yolov5s.pt")

        self.device = torch.device(device)
        base = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        base.classifier = nn.Linear(base.classifier[1].in_features, 2)

        self.classifier = base.to(self.device)
        if os.path.exists(CLASSIFY_PATH):
            self.classifier.load_state_dict(torch.load(CLASSIFY_PATH, map_location=self.device))

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.classifier.parameters(), lr=1e-4)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def save_models(self):
        torch.save(self.classifier.state_dict(), CLASSIFY_PATH)

    def train(self, video_paths, label):
        label_idx = 1 if label == "harmful" else 0
        all_frames, labels = [], []

        # Extract frames from all videos
        for path in video_paths:
            frames = extract_frames(path)
            for f in frames:
                all_frames.append(self.transform(f))
                labels.append(label_idx)

        if not all_frames:
            return

        # Convert to tensors
        data = torch.stack(all_frames)
        target = torch.tensor(labels, dtype=torch.long)

        # Split data into train and validation sets
        dataset = torch.utils.data.TensorDataset(data, target)
        dataset_size = len(dataset)
        val_size = int(0.2 * dataset_size)
        train_size = dataset_size - val_size

        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=8, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=8, shuffle=False
        )

        # Training loop with validation
        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_model_state = None

        for epoch in range(50):
            # Training phase
            self.classifier.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                out = self.classifier(x)
                loss = self.criterion(out, y)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item() * x.size(0)
                _, preds = torch.max(out, 1)
                train_correct += (preds == y).sum().item()
                train_total += y.size(0)

            train_loss = train_loss / train_total
            train_acc = train_correct / train_total

            # Validation phase
            self.classifier.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    out = self.classifier(x)
                    loss = self.criterion(out, y)

                    val_loss += loss.item() * x.size(0)
                    _, preds = torch.max(out, 1)
                    val_correct += (preds == y).sum().item()
                    val_total += y.size(0)

            val_loss = val_loss / val_total
            val_acc = val_correct / val_total

            # Print metrics
            print(f"Epoch {epoch + 1}/50:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            # Check if this is the best model so far
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                best_model_state = self.classifier.state_dict().copy()
            else:
                epochs_no_improve += 1

            # Early stopping
            if epochs_no_improve >= 5:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        # Restore best model
        if best_model_state is not None:
            self.classifier.load_state_dict(best_model_state)

        # Save models
        self.save_models()

        return {
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        }

    def predict(self, video_path):
        dets = self.detector.predict(source=video_path, verbose=False)

        for r in dets:
            for box in r.boxes:
                cls = self.detector.model.names[int(box.cls)]
                if cls in ["gun", "weapon"]:
                    return "harmful"

        frames = extract_frames(video_path)
        if not frames:
            return "harmless"

        self.classifier.eval()
        inputs = torch.stack([self.transform(f) for f in frames]).to(self.device)
        with torch.no_grad():
            logits = self.classifier(inputs)
            preds = logits.argmax(dim=1)

        if preds.sum().item() > len(preds) * 0.6:
            return "harmful"

        return "harmless"