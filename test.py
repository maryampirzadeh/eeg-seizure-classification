import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset import EEGDataset


DATASET_PATH = r"E:\Bonn Univeristy Dataset"
MODEL_PATH = "eeg_cnn.pth"


class SimpleEEGCNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(16, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(32),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 32, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


idx_to_class = {
    0: "Z",
    1: "O",
    2: "N",
    3: "F",
    4: "S",
}


def make_confusion_matrix(y_true, y_pred, num_classes=5):
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    dataset = EEGDataset(
        root_dir=DATASET_PATH,
        binary=False,
        normalize=True,
        add_channel_dim=True,
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    _, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    model = SimpleEEGCNN(num_classes=5).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    all_true = []
    all_pred = []

    print("\nSample predictions:\n")

    with torch.no_grad():
        for batch_idx, (signals, labels, class_names, file_paths) in enumerate(val_loader):
            signals = signals.to(device)
            labels = labels.to(device)

            outputs = model(signals)
            preds = outputs.argmax(dim=1)

            all_true.extend(labels.cpu().tolist())
            all_pred.extend(preds.cpu().tolist())

            if batch_idx == 0:
                for i in range(min(10, len(labels))):
                    true_label = idx_to_class[labels[i].item()]
                    pred_label = idx_to_class[preds[i].item()]
                    print(f"File: {file_paths[i]}")
                    print(f"True: {true_label} | Pred: {pred_label}")
                    print("-" * 50)

    all_true = torch.tensor(all_true)
    all_pred = torch.tensor(all_pred)

    accuracy = (all_true == all_pred).float().mean().item()
    cm = make_confusion_matrix(all_true, all_pred, num_classes=5)

    print(f"\nValidation Accuracy: {accuracy:.4f}\n")

    print("Confusion Matrix (rows=true, cols=pred):")
    print(cm)

    print("\nPer-class metrics:")
    for i in range(5):
        tp = cm[i, i].item()
        fp = cm[:, i].sum().item() - tp
        fn = cm[i, :].sum().item() - tp
        support = cm[i, :].sum().item()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)

        print(
            f"Class {idx_to_class[i]} | "
            f"Precision: {precision:.4f} | "
            f"Recall: {recall:.4f} | "
            f"Support: {support}"
        )


if __name__ == "__main__":
    main()