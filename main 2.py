import os
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

import torch
from torch.utils.data import DataLoader, random_split

from dataset import EEGDataset


# Put your path here if you want.
# Leave it as "" to choose the folder manually from a popup.
MANUAL_DATASET_PATH = r""


def choose_folder():
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title="Select the Bonn University Dataset folder")
    root.destroy()
    return folder


def resolve_dataset_root():
    """
    The folder you choose must directly contain:
    Z, O, N, F, S
    """
    if MANUAL_DATASET_PATH.strip():
        path = Path(MANUAL_DATASET_PATH)
    else:
        selected = choose_folder()
        if not selected:
            raise FileNotFoundError("No folder was selected.")
        path = Path(selected)

    print("\nSelected path:", path)
    print("Exists:", path.exists())

    if path.exists():
        try:
            print("Contents:", os.listdir(path))
        except Exception as e:
            print("Could not list folder contents:", e)

    required = ["Z", "O", "N", "F", "S"]
    missing = [x for x in required if not (path / x).is_dir()]
    if missing:
        raise FileNotFoundError(
            f"\nWrong folder selected: {path}\n"
            f"Missing subfolders: {missing}\n"
            f"You must select the folder that directly contains Z, O, N, F, S."
        )

    return path


def main():
    root_dir = resolve_dataset_root()

    dataset = EEGDataset(
        root_dir=root_dir,
        binary=False,          # True for seizure vs non-seizure
        normalize=True,
        add_channel_dim=True,
    )

    print("\nDataset loaded successfully")
    print("Total samples:", len(dataset))

    signal, label, class_name, file_path = dataset[0]
    print("\nSingle sample check")
    print("Signal shape:", signal.shape)
    print("Label:", label.item())
    print("Class:", class_name)
    print("File:", file_path)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
    )

    batch = next(iter(train_loader))
    signals, labels, class_names, file_paths = batch

    print("\nBatch check")
    print("Signals shape:", signals.shape)
    print("Labels shape:", labels.shape)
    print("First few labels:", labels[:8])
    print("First class:", class_names[0])
    print("First file:", file_paths[0])

    print("\nLoader works.")


if __name__ == "__main__":
    main()