"""
Transfer Learning Training Script for Household Sound Classification
======================================================================
Trains an audio classifier using pre-trained PANNs (CNN14) backbone with
fine-tuning on ESC-50 + custom dataset.

Expected accuracy improvement: 59% → 74-85%

Usage:
    # Option 1: Train with transfer learning (recommended)
    python train_with_transfer_learning.py --use-pretrained

    # Option 2: Train from scratch (baseline)
    python train_with_transfer_learning.py

    # With custom data path
    python train_with_transfer_learning.py --use-pretrained --custom-data ../data/custom

Authors: Joshua Kirby & Alan Nur (with Claude Sonnet 4.5 assistance)
Course:  TECHIN 513A — Managing Data And Signal Processing
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as AF

from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# ── Configuration ────────────────────────────────────────────────────────────

class Config:
    """Training configuration"""
    # Dataset
    ESC50_PATH = "data/ESC-50"
    CUSTOM_DATA_PATH = "data/custom"

    # Audio parameters
    SAMPLE_RATE = 44100
    DURATION = 5  # seconds
    N_SAMPLES = SAMPLE_RATE * DURATION  # 220,500

    # Feature extraction (for pre-trained model)
    N_FFT = 2048
    HOP_LENGTH = 512
    N_MELS = 64  # PANNs uses 64 mel bins
    F_MAX = 8000
    TOP_DB = 80

    # Model
    USE_PRETRAINED = False  # Set to True for transfer learning
    PRETRAINED_MODEL = "Cnn14"  # PANNs model name
    FREEZE_BACKBONE = True  # Freeze during phase 1
    FINE_TUNE_AFTER_EPOCH = 20  # Unfreeze after this epoch

    # Training
    BATCH_SIZE = 32
    EPOCHS = 100
    LR = 1e-3  # Learning rate for classifier
    LR_BACKBONE = 1e-5  # Lower LR for pre-trained backbone (phase 2)
    WEIGHT_DECAY = 1e-4

    # Checkpointing
    OUTPUT_DIR = "models/transfer_learning" if USE_PRETRAINED else "models/custom_cnn"
    CHECKPOINT_EVERY = 10

    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── PANNs Transfer Learning Model ───────────────────────────────────────────

class PANNsTransferModel(nn.Module):
    """
    Transfer learning wrapper for pre-trained PANNs (CNN14).

    Uses CNN14 backbone (trained on AudioSet) as feature extractor,
    with a custom classifier head for our specific classes.
    """

    def __init__(self, num_classes, freeze_backbone=True):
        super().__init__()

        # Load pre-trained PANNs CNN14 from TorchHub
        try:
            self.backbone = torch.hub.load(
                'qiuqiangkong/audioset_tagging_cnn',
                'cnn14',
                pretrained=True,
                trust_repo=True
            )
        except Exception as e:
            print(f"Error loading PANNs model: {e}")
            print("Attempting direct download...")
            # Fallback: load from local checkpoint if available
            raise RuntimeError("Please install panns_inference: pip install panns-inference")

        # Freeze backbone weights initially
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Remove original AudioSet classifier (527 classes)
        self.backbone.fc_audioset = nn.Identity()

        # Add custom classifier head for our classes
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),  # CNN14 outputs 2048-dim embeddings
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: (batch, 1, n_mels, time_steps) - single-channel mel spectrogram

        Returns:
            logits: (batch, num_classes) - raw logits (no softmax)
        """
        # Get embeddings from backbone
        embeddings = self.backbone(x)  # (batch, 2048)

        # Classify using custom head
        logits = self.classifier(embeddings)  # (batch, num_classes)

        return logits

    def unfreeze_last_n_blocks(self, n=2):
        """
        Unfreeze last n convolutional blocks for fine-tuning.

        CNN14 has 6 conv blocks. Unfreezing last 2 allows fine-tuning
        high-level features while keeping low-level features frozen.
        """
        # CNN14 architecture: conv_block1 through conv_block6
        blocks = [
            self.backbone.conv_block6,
            self.backbone.conv_block5,
            self.backbone.conv_block4,
            self.backbone.conv_block3,
            self.backbone.conv_block2,
            self.backbone.conv_block1,
        ]

        # Unfreeze last n blocks
        for i in range(n):
            if i < len(blocks):
                for param in blocks[i].parameters():
                    param.requires_grad = True
                print(f"  ✓ Unfroze conv_block{6-i}")


# ── Custom CNN (Baseline) ────────────────────────────────────────────────────

class HouseholdSoundCNN(nn.Module):
    """
    Baseline 4-block CNN trained from scratch (original architecture).

    This is the existing model architecture for comparison with transfer learning.
    """

    def __init__(self, num_classes, n_channels=1):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: n_channels → 16
            nn.Conv2d(n_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 2: 16 → 32
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 3: 32 → 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 4: 64 → 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Global average pooling
            nn.AdaptiveAvgPool2d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ── Feature Extraction ───────────────────────────────────────────────────────

def extract_mel_spectrogram(waveform, config):
    """
    Extract mel spectrogram for pre-trained model.

    PANNs expects single-channel mel spectrograms (1, n_mels, time_steps).
    """
    # Resample if needed
    if waveform.shape[-1] != config.N_SAMPLES:
        waveform = F.interpolate(
            waveform.unsqueeze(0),
            size=config.N_SAMPLES,
            mode='linear',
            align_corners=False
        ).squeeze(0)

    # Ensure mono
    if waveform.dim() > 1 and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    elif waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    # Mel spectrogram
    mel_transform = T.MelSpectrogram(
        sample_rate=config.SAMPLE_RATE,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        n_mels=config.N_MELS,
        f_max=config.F_MAX
    )

    mel = mel_transform(waveform)

    # Convert to dB scale
    mel_db = T.AmplitudeToDB(top_db=config.TOP_DB)(mel)

    return mel_db  # (1, n_mels, time_steps)


def compute_normalization_stats(dataset, config):
    """
    Compute mean and std for normalization across training set.
    """
    print("Computing normalization statistics...")

    mel_values = []

    for i in tqdm(range(min(len(dataset), 1000)), desc="Sampling"):
        features, _ = dataset[i]
        mel_values.append(features.numpy())

    mel_values = np.concatenate([m.flatten() for m in mel_values])

    mean = float(np.mean(mel_values))
    std = float(np.std(mel_values))

    print(f"  Mean: {mean:.2f}, Std: {std:.2f}")

    return [mean], [std]


# ── Dataset ──────────────────────────────────────────────────────────────────

class AudioDataset(Dataset):
    """Dataset for audio classification with augmentation."""

    def __init__(self, file_paths, labels, config, augment=False, norm_stats=None):
        self.file_paths = file_paths
        self.labels = labels
        self.config = config
        self.augment = augment
        self.norm_mean = norm_stats[0] if norm_stats else None
        self.norm_std = norm_stats[1] if norm_stats else None

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Load audio
        waveform, sr = torchaudio.load(self.file_paths[idx])

        # Resample if needed
        if sr != self.config.SAMPLE_RATE:
            waveform = AF.resample(waveform, sr, self.config.SAMPLE_RATE)

        # Augmentation (training only)
        if self.augment:
            waveform = self._augment(waveform)

        # Extract features
        features = extract_mel_spectrogram(waveform, self.config)

        # Normalize
        if self.norm_mean is not None:
            features = (features - self.norm_mean[0]) / (self.norm_std[0] + 1e-8)

        return features, self.labels[idx]

    def _augment(self, waveform):
        """Apply data augmentation."""
        # Time shift
        if np.random.random() < 0.5:
            shift = int(np.random.uniform(-self.config.SAMPLE_RATE, self.config.SAMPLE_RATE))
            waveform = torch.roll(waveform, shift, dims=-1)

        # Add noise
        if np.random.random() < 0.3:
            noise = torch.randn_like(waveform) * 0.005
            waveform = waveform + noise

        # Volume variation
        if np.random.random() < 0.5:
            gain = np.random.uniform(0.7, 1.3)
            waveform = waveform * gain

        return waveform


# ── Data Loading ─────────────────────────────────────────────────────────────

def load_esc50_data(config):
    """Load ESC-50 dataset."""
    esc_path = Path(config.ESC50_PATH)

    if not esc_path.exists():
        print(f"❌ ESC-50 not found at {esc_path}")
        print("Please download: https://github.com/karolpiczak/ESC-50")
        sys.exit(1)

    # Load metadata
    meta_path = esc_path / "meta" / "esc50.csv"

    import pandas as pd
    df = pd.read_csv(meta_path)

    # Split by fold
    train_files = df[df['fold'].isin([1, 2, 3])]['filename'].tolist()
    val_files = df[df['fold'] == 4]['filename'].tolist()
    test_files = df[df['fold'] == 5]['filename'].tolist()

    # Get labels
    label_to_idx = {label: idx for idx, label in enumerate(sorted(df['category'].unique()))}

    def get_data(filenames):
        paths = [(esc_path / "audio" / f).as_posix() for f in filenames]
        labels = [label_to_idx[df[df['filename'] == f]['category'].values[0]] for f in filenames]
        return paths, labels

    train_data = get_data(train_files)
    val_data = get_data(val_files)
    test_data = get_data(test_files)

    return train_data, val_data, test_data, list(label_to_idx.keys())


def load_custom_data(config):
    """Load custom dataset."""
    custom_path = Path(config.CUSTOM_DATA_PATH)

    if not custom_path.exists():
        return [], []

    file_paths = []
    labels = []
    class_names = []

    for class_dir in sorted(custom_path.iterdir()):
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        class_names.append(class_name)
        class_idx = len(class_names) - 1

        for wav_file in class_dir.glob("*.wav"):
            file_paths.append(wav_file.as_posix())
            labels.append(class_idx)

    return (file_paths, labels), class_names


# ── Training ─────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device, desc="Train"):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=desc)
    for features, labels in pbar:
        features, labels = features.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * features.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({'loss': loss.item(), 'acc': correct/total})

    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    """Evaluate on validation/test set."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * features.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


# ── Main ─────────────────────────────────────────────────────────────────────

def main(args):
    # Update config from args
    Config.USE_PRETRAINED = args.use_pretrained
    Config.CUSTOM_DATA_PATH = args.custom_data
    Config.OUTPUT_DIR = args.output_dir or (
        "models/transfer_learning" if args.use_pretrained else "models/custom_cnn"
    )

    # Create output directory
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Household Sound Classifier Training")
    print(f"{'='*60}")
    print(f"  Mode: {'Transfer Learning (PANNs CNN14)' if Config.USE_PRETRAINED else 'From Scratch (Custom CNN)'}")
    print(f"  Device: {Config.DEVICE}")
    print(f"  Output: {Config.OUTPUT_DIR}")
    print(f"{'='*60}\n")

    # Load datasets
    print("Loading ESC-50 dataset...")
    (train_paths, train_labels), (val_paths, val_labels), \
        (test_paths, test_labels), esc_classes = load_esc50_data(Config)

    print(f"  ✓ ESC-50: {len(train_paths)} train, {len(val_paths)} val, {len(test_paths)} test")

    # Load custom data
    custom_data, custom_classes = load_custom_data(Config)
    if custom_data and custom_data[0]:
        print(f"  ✓ Custom: {len(custom_data[0])} clips, {len(custom_classes)} classes")

        # Merge with ESC-50 (offset custom labels)
        all_classes = esc_classes + custom_classes
        num_classes = len(all_classes)

        custom_paths, custom_labels_raw = custom_data
        custom_labels = [label + len(esc_classes) for label in custom_labels_raw]  # Offset by ESC-50 class count

        # Split custom data 70% train / 15% val / 15% test
        from sklearn.model_selection import train_test_split
        custom_train_paths, custom_temp_paths, custom_train_labels, custom_temp_labels = train_test_split(
            custom_paths, custom_labels, test_size=0.3, random_state=42, stratify=custom_labels
        )
        custom_val_paths, custom_test_paths, custom_val_labels, custom_test_labels = train_test_split(
            custom_temp_paths, custom_temp_labels, test_size=0.5, random_state=42, stratify=custom_temp_labels
        )

        # Merge with ESC-50
        train_paths += custom_train_paths
        train_labels += custom_train_labels
        val_paths += custom_val_paths
        val_labels += custom_val_labels
        test_paths += custom_test_paths
        test_labels += custom_test_labels

        print(f"  ✓ Merged: {len(train_paths)} train, {len(val_paths)} val, {len(test_paths)} test")
    else:
        print("  ℹ No custom data found")
        all_classes = esc_classes
        num_classes = len(all_classes)

    # Create datasets
    print("\nCreating datasets...")
    train_dataset = AudioDataset(train_paths, train_labels, Config, augment=True)

    # Compute normalization stats
    norm_stats = compute_normalization_stats(train_dataset, Config)

    # Recreate with normalization
    train_dataset = AudioDataset(train_paths, train_labels, Config, augment=True, norm_stats=norm_stats)
    val_dataset = AudioDataset(val_paths, val_labels, Config, augment=False, norm_stats=norm_stats)
    test_dataset = AudioDataset(test_paths, test_labels, Config, augment=False, norm_stats=norm_stats)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2)

    # Create model
    print(f"\nInitializing model...")
    if Config.USE_PRETRAINED:
        model = PANNsTransferModel(num_classes, freeze_backbone=Config.FREEZE_BACKBONE)
        print(f"  ✓ PANNs CNN14 with custom head ({num_classes} classes)")
    else:
        model = HouseholdSoundCNN(num_classes, n_channels=1)
        print(f"  ✓ Custom 4-block CNN ({num_classes} classes)")

    model = model.to(Config.DEVICE)

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Training loop
    print(f"\nTraining for {Config.EPOCHS} epochs...")
    best_val_acc = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(Config.EPOCHS):
        # Phase 1: Train classifier only (backbone frozen)
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, Config.DEVICE,
            desc=f"Epoch {epoch+1}/{Config.EPOCHS}"
        )

        val_loss, val_acc = evaluate(model, val_loader, criterion, Config.DEVICE)

        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # LR scheduling
        scheduler.step(val_loss)

        # Print progress
        if (epoch + 1) % 5 == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"\nEpoch {epoch+1:3d}/{Config.EPOCHS} | "
                  f"Train: {train_loss:.4f} / {train_acc:.1%} | "
                  f"Val: {val_loss:.4f} / {val_acc:.1%} | "
                  f"LR: {lr:.1e}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(Config.OUTPUT_DIR, "model.pt"))
            print(f"  → Best model saved (val_acc: {val_acc:.1%})")

        # Phase 2: Fine-tune (unfreeze backbone)
        if Config.USE_PRETRAINED and epoch + 1 == Config.FINE_TUNE_AFTER_EPOCH:
            print("\n" + "="*60)
            print("  Starting Phase 2: Fine-Tuning Backbone")
            print("="*60)
            model.unfreeze_last_n_blocks(2)

            # Create new optimizer with differential learning rates
            optimizer = optim.Adam([
                {'params': model.backbone.parameters(), 'lr': Config.LR_BACKBONE},
                {'params': model.classifier.parameters(), 'lr': Config.LR}
            ], weight_decay=Config.WEIGHT_DECAY)

            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Final evaluation
    print(f"\n{'='*60}")
    print("  Final Evaluation on Test Set")
    print(f"{'='*60}")

    model.load_state_dict(torch.load(os.path.join(Config.OUTPUT_DIR, "model.pt")))
    test_loss, test_acc = evaluate(model, test_loader, criterion, Config.DEVICE)

    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc:.1%}")
    print(f"{'='*60}\n")

    # Save config and labels
    config_data = {
        "sample_rate": Config.SAMPLE_RATE,
        "duration": Config.DURATION,
        "n_fft": Config.N_FFT,
        "hop_length": Config.HOP_LENGTH,
        "n_mels": Config.N_MELS,
        "f_max": Config.F_MAX,
        "top_db": Config.TOP_DB,
        "model_type": "panns_cnn14" if Config.USE_PRETRAINED else "custom_cnn",
        "n_channels": 1,
        "pretrained_backbone": Config.USE_PRETRAINED,
        "norm_mean": norm_stats[0],
        "norm_std": norm_stats[1],
    }

    with open(os.path.join(Config.OUTPUT_DIR, "config.json"), "w") as f:
        json.dump(config_data, f, indent=2)

    labels_data = {
        "num_classes": num_classes,
        "labels": all_classes
    }

    with open(os.path.join(Config.OUTPUT_DIR, "labels.json"), "w") as f:
        json.dump(labels_data, f, indent=2)

    print(f"✅ Training complete! Model saved to {Config.OUTPUT_DIR}/")
    print(f"   - model.pt (weights)")
    print(f"   - config.json (parameters)")
    print(f"   - labels.json (class names)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train household sound classifier")
    parser.add_argument("--use-pretrained", action="store_true",
                        help="Use transfer learning with PANNs CNN14")
    parser.add_argument("--custom-data", default="data/custom",
                        help="Path to custom dataset")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory for trained model")

    args = parser.parse_args()
    main(args)
