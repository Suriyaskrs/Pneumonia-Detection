import torch
import torch.nn as nn
import timm

# -----------------------------
# CNN HEAD (Exact Training Version)
# -----------------------------
class CNNHead(nn.Module):
    def __init__(self):
        super().__init__()

        # 768 → 3x16x16
        self.fc_to_map = nn.Linear(768, 3 * 16 * 16)

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.fc_to_map(x)
        x = x.view(-1, 3, 16, 16)
        x = self.conv(x)
        return self.classifier(x)


# -----------------------------
# HYBRID MODEL (Exact Training Version)
# -----------------------------
class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()

        # SAME as training
        self.vit = timm.create_model(
            "vit_base_patch16_224",
            pretrained=False  # IMPORTANT: False for loading weights
        )

        self.vit.head = nn.Identity()

        # freeze vit (like training)
        for p in self.vit.parameters():
            p.requires_grad = False

        self.cnn = CNNHead()

    def forward(self, x):
        features = self.vit(x)
        return self.cnn(features)


# -----------------------------
# MODEL LOADER
# -----------------------------
def load_model(weights_path: str, device="cpu"):
    model = HybridModel()
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model