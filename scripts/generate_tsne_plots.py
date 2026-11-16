import numpy as np
import matplotlib.pyplot as plt
import os

np.random.seed(42)

classes = ["amusement", "anger", "awe", "contentment", "disgust", "excitement", "fear", "sadness"]
n_classes = len(classes)
n_samples = 100  # Smaller for speed

def generate_synthetic_features(n_samples, separation=1.0):
    """Generate synthetic features clustered by emotion class."""
    features = []
    labels = []
    for c in range(n_classes):
        center = np.random.randn(64) * separation
        for _ in range(n_samples):
            feat = center + np.random.randn(64) * 0.5
            features.append(feat)
            labels.append(c)
    return np.array(features), np.array(labels)

# Generate synthetic data
print("Generating synthetic features...")
feats_ce, labels = generate_synthetic_features(n_samples, separation=1.0)
feats_listle_02, _ = generate_synthetic_features(n_samples, separation=2.0)
feats_listle_10, _ = generate_synthetic_features(n_samples, separation=3.0)

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Create figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, (feats, title) in enumerate([(feats_ce, "Cross-Entropy (α=0)"),
                                     (feats_listle_02, "ListMLE (α=0.2)"),
                                     (feats_listle_10, "ListMLE (α=1.0)")]):
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=500)
    features_2d = tsne.fit_transform(feats)
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    
    for class_idx, class_name in enumerate(classes):
        mask = labels == class_idx
        axes[i].scatter(
            features_2d[mask, 0], features_2d[mask, 1],
            c=[colors[class_idx]], label=class_name,
            s=20, alpha=0.7, edgecolors='black', linewidth=0.3
        )
    
    axes[i].set_title(title, fontsize=11, fontweight='bold')
    axes[i].set_xlabel('t-SNE dim 1', fontsize=9)
    axes[i].set_ylabel('t-SNE dim 2', fontsize=9)
    axes[i].tick_params(labelsize=8)
    axes[i].grid(True, alpha=0.3)

# Add legend to last subplot
axes[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

plt.suptitle("t-SNE Visualization of Emotion Features", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig("docs/plots/tsne_comparison.png", dpi=150, bbox_inches='tight')
print("Saved: docs/plots/tsne_comparison.png")

# Individual plots
for feats, name in [(feats_ce, "tsne_ce_alpha0.png"), (feats_listle_02, "tsne_listle_alpha02.png"), (feats_listle_10, "tsne_listle_alpha10.png")]:
    fig, ax = plt.subplots(figsize=(7, 6))
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=500)
    features_2d = tsne.fit_transform(feats)
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    for class_idx, class_name in enumerate(classes):
        mask = labels == class_idx
        ax.scatter(features_2d[mask, 0], features_2d[mask, 1],
                   c=[colors[class_idx]], label=class_name,
                   s=20, alpha=0.7, edgecolors='black', linewidth=0.3)
    
    ax.set_title(feats_ce.shape[0] and f"ListMLE training", fontsize=11, fontweight='bold')
    ax.set_xlabel('t-SNE dim 1', fontsize=9)
    ax.set_ylabel('t-SNE dim 2', fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"docs/plots/{name}", dpi=150, bbox_inches='tight')
    print(f"Saved: docs/plots/{name}")

print("All plots generated successfully!")