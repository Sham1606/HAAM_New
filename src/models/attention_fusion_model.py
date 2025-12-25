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
    def __init__(self, acoustic_dim=12, text_dim=768, num_classes=5, hidden_dim=128, nhead=4):
        super().__init__()
        
        # Branch Projection Layers
        self.acoustic_proj = nn.Sequential(
            nn.Linear(acoustic_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Interaction Layer: Multi-Head Attention
        # We treat acoustic and text as 2 tokens in a sequence
        self.mha = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=nhead, dropout=0.1, batch_first=True)
        
        # Modality Tokens (Learnable)
        self.modality_pos = nn.Parameter(torch.randn(1, 2, hidden_dim))
        
        # Gating Layer for Fusion Weights (Dashboard Compatibility)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=1)
        )
        
        # Final Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, acoustic, text):
        # 1. Project to common hidden space
        a_feat = self.acoustic_proj(acoustic).unsqueeze(1) # [batch, 1, hidden]
        t_feat = self.text_proj(text).unsqueeze(1)         # [batch, 1, hidden]
        
        # 2. Concat into sequence
        # x shape: [batch, 2, hidden]
        x = torch.cat([a_feat, t_feat], dim=1)
        
        # 3. Add modality-specific positional encoding
        x = x + self.modality_pos
        
        # 4. Multi-Head Interaction Attention
        # attn_output: [batch, 2, hidden]
        # attn_weights: [batch, 2, 2]
        attn_out, _ = self.mha(x, x, x)
        
        # 5. Extract interacted features
        a_inter = attn_out[:, 0, :]
        t_inter = attn_out[:, 1, :]
        
        # 6. Calculate Gating Weights for Fusion (Dashboard)
        combined_aux = torch.cat([a_inter, t_inter], dim=1) # [batch, hidden*2]
        weights = self.gate(combined_aux) # [batch, 2]
        
        # 7. Weighted Fusion
        fused = (weights[:, 0].unsqueeze(1) * a_inter) + (weights[:, 1].unsqueeze(1) * t_inter)
        
        # 8. Classify
        logits = self.classifier(fused)
        
        return logits, weights
