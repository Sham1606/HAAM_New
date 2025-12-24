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
    def __init__(self, acoustic_dim=43, text_dim=768, num_classes=5, hidden_dim=256, nhead=8):
        super().__init__()
        
        # Branch Projection Layers
        self.acoustic_proj = nn.Sequential(
            nn.Linear(acoustic_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(0.2)
        )
        
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(0.2)
        )
        
        # Interaction Layer: Multi-Head Attention
        self.mha = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=nhead, dropout=0.2, batch_first=True)
        self.norm_mha = nn.LayerNorm(hidden_dim)
        
        # Modality Tokens (Learnable)
        self.modality_pos = nn.Parameter(torch.randn(1, 2, hidden_dim))
        
        # Gating Layer for Fusion Weights
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=1)
        )
        
        # Final Classifier with Residual-like depth
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, acoustic, text):
        # 1. Project to common hidden space
        a_feat = self.acoustic_proj(acoustic).unsqueeze(1) # [batch, 1, hidden]
        t_feat = self.text_proj(text).unsqueeze(1)         # [batch, 1, hidden]
        
        # 2. Concat into sequence
        x = torch.cat([a_feat, t_feat], dim=1)
        
        # 3. Add modality-specific positional encoding
        x = x + self.modality_pos
        
        # 4. Multi-Head Interaction Attention (with Residual Connection)
        attn_out, _ = self.mha(x, x, x)
        x = self.norm_mha(x + attn_out) # Residual + Norm
        
        # 5. Extract interacted features
        a_inter = x[:, 0, :]
        t_inter = x[:, 1, :]
        
        # 6. Calculate Gating Weights for Fusion
        combined_aux = torch.cat([a_inter, t_inter], dim=1)
        weights = self.gate(combined_aux)
        
        # 7. Weighted Fusion
        fused = (weights[:, 0].unsqueeze(1) * a_inter) + (weights[:, 1].unsqueeze(1) * t_inter)
        
        # 8. Classify
        logits = self.classifier(fused)
        
        return logits, weights
