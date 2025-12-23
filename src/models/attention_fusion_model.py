"""
Improved model architecture with Attention Fusion
Inputs:
- Acoustic: 12 features
- Text: 768 dim embedding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionFusionNetwork(nn.Module):
    def __init__(self, acoustic_dim=12, text_dim=768, num_classes=5, hidden_dim=128):
        super().__init__()
        
        # Acoustic Branch
        self.acoustic_net = nn.Sequential(
            nn.Linear(acoustic_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Text Branch
        self.text_net = nn.Sequential(
            nn.Linear(text_dim, hidden_dim*2),
            nn.LayerNorm(hidden_dim*2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim*2, hidden_dim)
        )
        
        # Attention Fusion
        # Calculate attention weights for each modality
        self.attention_fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 2), # 2 weights (acoustic, text)
            nn.Softmax(dim=1)
        )
        
        # Combined Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, acoustic, text):
        a_feat = self.acoustic_net(acoustic) # [batch, hidden]
        t_feat = self.text_net(text)         # [batch, hidden]
        
        # Concat for attention
        combined = torch.cat([a_feat, t_feat], dim=1) # [batch, hidden*2]
        
        # Get weights
        weights = self.attention_fc(combined) # [batch, 2]
        
        # Weighted sum: w0*a + w1*t
        # Expand weights for broadcasting
        w_a = weights[:, 0].unsqueeze(1)
        w_t = weights[:, 1].unsqueeze(1)
        
        fused = (w_a * a_feat) + (w_t * t_feat)
        
        output = self.classifier(fused)
        
        return output, weights
