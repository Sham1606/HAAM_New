import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block: output = activation(Linear(x)) + projection(x)"""
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.25):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
        )
        self.skip = nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else nn.Identity()
        self.act  = nn.GELU()

    def forward(self, x):
        return self.act(self.main(x) + self.skip(x))


class CrossModalAttention(nn.Module):
    """
    Scaled dot-product attention: query from one modality, key/value from the other.
    Learns how acoustic and text features should cross-attend to each other.
    """
    def __init__(self, q_dim: int, kv_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.scale    = out_dim ** -0.5
        self.q_proj   = nn.Linear(q_dim,  out_dim, bias=False)
        self.k_proj   = nn.Linear(kv_dim, out_dim, bias=False)
        self.v_proj   = nn.Linear(kv_dim, out_dim, bias=False)
        self.out_proj = nn.Linear(out_dim, out_dim)
        self.dropout  = nn.Dropout(dropout)

    def forward(self, query, key_value):
        Q = self.q_proj(query)                        # [B, D]
        K = self.k_proj(key_value)                    # [B, D]
        V = self.v_proj(key_value)                    # [B, D]
        # Single-vector attention (batch of scalars)
        attn = torch.sigmoid((Q * K).sum(dim=-1, keepdim=True) * self.scale)   # [B, 1]
        out  = self.dropout(attn) * V                 # [B, D]
        return self.out_proj(out), attn.squeeze(-1)   # [B, D], [B]


class ImprovedHybridModel(nn.Module):
    """
    Enhanced Hybrid Fusion Model v2.1

    Architecture:
    ┌──────────────────────────────┐    ┌──────────────────────────────┐
    │  Acoustic Branch             │    │  Text Branch                 │
    │  20 → 128 → 64 → 64         │    │  773 → 512 → 256 → 256       │
    │  (3 residual blocks + GELU)  │    │  (3 residual blocks + GELU)  │
    └──────────────┬───────────────┘    └──────────────┬───────────────┘
                   │                                   │
                   ├──── Cross-Modal Attention ←───────┘
                   │       (acoustic attends to text,
                   │        text attends to acoustic)
                   ↓
           concat [a_attended(64) || t_attended(256)] = 320
                   ↓
           Attention Gate → [w_audio, w_text]   ← XAI output
                   ↓
           Classifier (320 → 128 → n_classes)
    """

    def __init__(self, n_acoustic=20, n_text_emb=768, n_text_probs=5,
                 n_classes=5, dropout=0.25):
        super().__init__()

        A_DIM = 64    # final acoustic branch dim
        T_DIM = 256   # final text branch dim
        FUSED = A_DIM + T_DIM  # 320

        # ── Acoustic Branch: 20 → 128 → 64 → 64 ──────────────────────────────
        self.acoustic_input = nn.Sequential(
            nn.Linear(n_acoustic, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.acoustic_res1 = ResidualBlock(128, 64, dropout=dropout)
        self.acoustic_res2 = ResidualBlock(64,  A_DIM, dropout=dropout)

        # ── Text Branch: (768+5) → 512 → 256 → 256 ───────────────────────────
        self.text_input_dim = n_text_emb + n_text_probs           # 773
        self.text_input = nn.Sequential(
            nn.Linear(self.text_input_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.text_res1 = ResidualBlock(512, 256, dropout=dropout)
        self.text_res2 = ResidualBlock(256, T_DIM, dropout=dropout)

        # ── Cross-Modal Attention ──────────────────────────────────────────────
        # acoustic (Q) attends to text (K,V) → a_attended [B, A_DIM]
        self.a2t_attn = CrossModalAttention(A_DIM, T_DIM, A_DIM, dropout=0.1)
        # text (Q) attends to acoustic (K,V) → t_attended [B, T_DIM]
        self.t2a_attn = CrossModalAttention(T_DIM, A_DIM, T_DIM, dropout=0.1)

        # ── Attention Gate (XAI) ──────────────────────────────────────────────
        self.attention_gate = nn.Sequential(
            nn.Linear(FUSED, 64),
            nn.Tanh(),
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        )

        # ── Classification Head: 320 → 128 → n_classes ───────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(FUSED, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x_acoustic, x_text_emb, x_text_probs):
        """
        Args:
            x_acoustic:   [B, 20]
            x_text_emb:   [B, 768]
            x_text_probs: [B, 5]

        Returns:
            logits:       [B, n_classes]
            attn_weights: [B, 2]  → [w_audio, w_text]  for XAI
        """
        # 1. Acoustic Branch
        a = self.acoustic_input(x_acoustic)    # [B, 128]
        a = self.acoustic_res1(a)               # [B, 64]
        a_out = self.acoustic_res2(a)           # [B, 64]

        # 2. Text Branch
        t_input = torch.cat([x_text_emb, x_text_probs], dim=1)   # [B, 773]
        t = self.text_input(t_input)            # [B, 512]
        t = self.text_res1(t)                   # [B, 256]
        t_out = self.text_res2(t)               # [B, 256]

        # 3. Cross-Modal Attention
        a_attended, _  = self.a2t_attn(a_out, t_out)   # acoustic attends to text
        t_attended, _  = self.t2a_attn(t_out, a_out)   # text attends to acoustic

        # Residual connection: add original + attended
        a_final = a_out + a_attended    # [B, 64]
        t_final = t_out + t_attended    # [B, 256]

        # 4. Fuse
        combined = torch.cat([a_final, t_final], dim=1)  # [B, 320]

        # 5. Attention Gate (XAI)
        attn_weights = self.attention_gate(combined)      # [B, 2]

        # 6. Classify
        logits = self.classifier(combined)                # [B, n_classes]

        return logits, attn_weights
