import torch
import torch.nn as nn

class HybridFusionNetwork(nn.Module):
    def __init__(self, n_acoustic=3, n_text=5, n_classes=5, hidden_dim=64):
        super(HybridFusionNetwork, self).__init__()
        
        # Acoustic Branch
        self.acoustic_net = nn.Sequential(
            nn.Linear(n_acoustic, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Text/Sentiment Branch
        self.text_net = nn.Sequential(
            nn.Linear(n_text, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Fusion Layer (Attention-based)
        self.fusion_dim = hidden_dim + (hidden_dim // 2)
        self.attention = nn.Sequential(
            nn.Linear(self.fusion_dim, self.fusion_dim),
            nn.Tanh(),
            nn.Linear(self.fusion_dim, 1),
            nn.Softmax(dim=1)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, n_classes)
        )

    def forward(self, x_acoustic, x_text):
        out_a = self.acoustic_net(x_acoustic)
        out_t = self.text_net(x_text)
        
        # Concatenate
        combined = torch.cat((out_a, out_t), dim=1)
        
        # Attention mechanism (Not fully utilized in this simple concat, but present in architecture)
        # To strictly match training script, we just feed 'combined' to classifier if training script did that.
        # Let's check training script implementation to be EXACT.
        # Training script: 
        # combined = torch.cat((out_a, out_t), dim=1)
        # out = self.classifier(combined)
        # The 'self.attention' layer was defined but NOT USED in forward(). 
        # To match weights, I must define it so state_dict loads, but forward must skip it like training did.
        
        out = self.classifier(combined)
        return out
