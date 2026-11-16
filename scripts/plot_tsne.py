"""
t-SNE visualization of emotion features from trained ECC/EMC models.

Extracts penultimate layer features from a trained checkpoint and visualizes
them using t-SNE, colored by emotion class.

Usage:
    python scripts/plot_tsne.py \
        --checkpoint runs/emoset_r50_a0.2/best.pt \
        --config configs/emoset_resnet50.yaml \
        --data-root data/EmoSet \
        --data-splits data/EmoSet_splits \
        --split val \
        --output docs/plots/tsne_emoset_a0.2.png

    # Compare CE vs ListMLE side by side:
    python scripts/plot_tsne.py --checkpoint runs/emoset_r50_a0/best.pt \
        --config configs/emoset_resnet50.yaml --set loss.alpha=0.0 \
        --data-root data/EmoSet --data-splits data/EmoSet_splits \
        --output docs/plots/tsne_emoset_alpha0.png

    python scripts/plot_tsne.py --config configs/emoset_resnet50.yaml \
        --set loss.alpha=0.2 \
        --data-root data/EmoSet --data-splits data/EmoSet_splits \
        --output docs/plots/tsne_emoset_alpha0.2.png
"""

import argparse
import json
import torch
from pathlib import Path
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from ecc.models import build_model
from ecc.data import EmotionCSV, build_transforms
from ecc.emotion import polarity_vector


def extract_features(model, loader, device):
    """Extract penultimate layer features and labels from model."""
    model.eval()
    features = []
    labels = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            # Get features before final classification layer
            feats = model(x)
            # Remove final fc layer output - use penultimate features
            # ResNet fc layer is the last layer; we want features before it
            # We'll capture via forward hook on the fc layer input
            features.append(feats.cpu())
            labels.append(y.cpu())

    features = torch.cat(features, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()
    return features, labels


def extract_features_with_hook(model, loader, device):
    """Extract penultimate layer features using forward hook on fc layer."""
    model.eval()
    features = []
    labels = []

    features_list = []

    def hook_fn(module, input, output):
        # input[0] is the feature vector before the fc layer
        features_list.append(input[0].detach().cpu())

    handle = model.fc.register_forward_hook(hook_fn)

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            _ = model(x)

        features.append(features_list[-1])  # last batch features
        # Actually we need to collect all batches
        # Let's restructure

    # Re-do properly
    features_list = []

    def hook_fn2(module, input, output):
        features_list.append(input[0].detach().cpu())

    handle = model.fc.register_forward_hook(hook_fn2)

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            _ = model(x)
            labels.append(y.cpu())

    handle.remove()

    features = torch.cat(features_list, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()
    return features, labels


def plot_tsne(features, labels, class_names, title, output_path):
    """Create t-SNE visualization."""
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    features_2d = tsne.fit_transform(features)

    fig, ax = plt.subplots(figsize=(10, 8))

    n_classes = len(class_names)
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))

    for class_idx, class_name in enumerate(classes):
        mask = labels == class_idx
        ax.scatter(
            features_2d[mask, 0],
            features_2d[mask, 1],
            c=[colors[class_idx]],
            label=class_name,
            s=30,
            alpha=0.7,
            edgecolors='black',
            linewidth=0.5
        )

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('t-SNE dimension 1')
    ax.set_ylabel('t-SNE dimension 2')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='t-SNE visualization of ECC/EMC features')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pt file)')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config YAML file')
    parser.add_argument('--set', nargs='*', default=[],
                        help='Override config values (e.g. --set loss.alpha=0.2)')
    parser.add_argument('--data-root', type=str, default='data/EmoSet',
                        help='Path to EmoSet image root')
    parser.add_argument('--data-splits', type=str, default='data/EmoSet_splits',
                        help='Path to splits directory')
    parser.add_argument('--split', type=str, default='val',
                        help='Split to visualize (train/val/test)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for feature extraction')
    parser.add_argument('--output', type=str, default='docs/plots/tsne_emoset.png',
                        help='Output path for t-SNE plot')
    args = parser.parse_args()

    import yaml
    from ecc.config import load_config

    # Load config
    cfg = load_config(args.config, args.set)

    # Build model
    classes = sorted(['amusement', 'anger', 'awe', 'contentment', 'disgust', 'excitement', 'fear', 'sadness'])
    model = build_model(cfg["model"]["name"], len(classes), cfg["model"]["pretrained"])
    
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model'] if 'model' in checkpoint else checkpoint)
    model.eval()

    # Dataset
    ds = EmotionCSV(
        Path(args.splits_dir) / f"{args.split}.csv",
        args.data_root,
        classes,
        build_transforms(False, cfg["data"]["image_size"], cfg["data"]["resize"])
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Extract features
    features, labels = extract_features_with_hook(model, loader, device)

    # Plot
    plot_tsne(features, labels, classes,
              f"t-SNE: {args.config} (alpha={cfg['loss']['alpha']})",
              args.output)


if __name__ == "__main__":
    main()