import torch
import torch.nn as nn
import timm

# -----------------------------
# CNN HEAD (same as training)
# -----------------------------
class CNNHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(768, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.fc = nn.Linear(128, 2)

    def forward(self, x):
        # x shape from ViT = (batch, 768)
        x = x.unsqueeze(2)            # → (batch, 768, 1)
        x = self.conv(x)              # → (batch, 128, 1)
        x = x.squeeze(2)              # → (batch, 128)
        x = self.fc(x)                # → (batch, 2)
        return x


# -----------------------------
# HYBRID VIT + CNN MODEL
# -----------------------------
class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Vision Transformer backbone
        self.vit = timm.create_model(
            "vit_base_patch16_224",
            pretrained=False,
            num_classes=0   # IMPORTANT → outputs embeddings
        )

        # CNN classifier head
        self.cnn = CNNHead()

    def forward(self, x):
        features = self.vit(x)  # → (batch, 768)
        out = self.cnn(features)
        return out


# -----------------------------
# MODEL LOADER FUNCTION
# -----------------------------
def load_model(weights_path: str, device="cpu"):
    model = HybridModel()
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model