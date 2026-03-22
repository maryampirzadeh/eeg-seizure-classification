from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


class EEGDataset(Dataset):
    def __init__(
        self,
        root_dir,
        binary=False,
        normalize=True,
        add_channel_dim=True,
        extensions=(".txt", ".TXT"),
    ):
        self.root_dir = Path(root_dir)
        self.binary = binary
        self.normalize = normalize
        self.add_channel_dim = add_channel_dim
        self.extensions = extensions

        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset folder not found: {self.root_dir}")

        required = ["Z", "O", "N", "F", "S"]
        missing = [x for x in required if not (self.root_dir / x).is_dir()]
        if missing:
            raise FileNotFoundError(
                f"Missing folders {missing} inside {self.root_dir}\n"
                f"The selected folder must directly contain Z, O, N, F, S"
            )

        if binary:
            self.label_map = {"Z": 0, "O": 0, "N": 0, "F": 0, "S": 1}
        else:
            self.label_map = {"Z": 0, "O": 1, "N": 2, "F": 3, "S": 4}

        self.samples = self._build_index()

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No EEG files found in {self.root_dir}. "
                f"Check that the files are .txt and inside Z/O/N/F/S."
            )

    def _build_index(self):
        samples = []

        for class_name in ["Z", "O", "N", "F", "S"]:
            class_dir = self.root_dir / class_name

            files = []
            for ext in self.extensions:
                files.extend(sorted(class_dir.glob(f"*{ext}")))

            for file_path in files:
                samples.append((file_path, self.label_map[class_name], class_name))

        return samples

    def __len__(self):
        return len(self.samples)

    def _load_signal(self, file_path):
        signal = np.loadtxt(file_path, dtype=np.float32)

        if signal.ndim > 1:
            signal = signal.reshape(-1)

        if self.normalize:
            mean = signal.mean()
            std = signal.std()
            signal = (signal - mean) / (std + 1e-8)

        return signal

    def __getitem__(self, idx):
        file_path, label, class_name = self.samples[idx]

        signal = self._load_signal(file_path)
        signal = torch.tensor(signal, dtype=torch.float32)

        if self.add_channel_dim:
            signal = signal.unsqueeze(0)  # (1, L)

        label = torch.tensor(label, dtype=torch.long)
        return signal, label, class_name, str(file_path)