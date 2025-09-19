import torch
import torch.nn as nn
import torch.nn.functional as F

class MultimodalFusionModel(nn.Module):
    """
    Multimodal fusion with cross-attention + gated fusion + audio-bias.
    Replaces previous fusion model. Works with audio=768, text=768 by default.
    """

    def __init__(self,
                 audio_input_dim=768,
                 text_input_dim=768,
                 hidden_dim=1024,
                 num_sentiment_classes=3,
                 num_emotion_classes=7,
                 dropout_rate=0.3,
                 n_heads=4):
        super().__init__()

        # Projections into hidden dim
        self.audio_proj_in = nn.Linear(audio_input_dim, hidden_dim)
        self.text_proj_in = nn.Linear(text_input_dim, hidden_dim)

        # LayerNorms
        self.norm_audio = nn.LayerNorm(hidden_dim)
        self.norm_text = nn.LayerNorm(hidden_dim)

        # Cross-attention modules (batch_first=True)
        self.cross_attn_audio2text = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=n_heads, batch_first=True)
        self.cross_attn_text2audio = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=n_heads, batch_first=True)

        # Gating (GMU-style)
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.gate_act = nn.Sigmoid()

        # Fusion attention score and small learnable audio bias
        self.attn_score = nn.Linear(hidden_dim * 2, 2)
        self.audio_bias = nn.Parameter(torch.tensor(0.05), requires_grad=True)

        # Shared MLP
        self.shared_ln = nn.LayerNorm(hidden_dim)
        self.mlp1 = nn.Linear(hidden_dim, hidden_dim)
        self.mlp2 = nn.Linear(hidden_dim, hidden_dim)

        # Classifiers
        self.sentiment_head = nn.Linear(hidden_dim, num_sentiment_classes)
        self.emotion_head = nn.Linear(hidden_dim, num_emotion_classes)

        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

def forward(self, audio_embeddings, text_embeddings, mode="fusion"):
    """
    mode: "fusion", "audio", "text"
    """
    # Project
    a = self.relu(self.audio_proj_in(audio_embeddings))
    t = self.relu(self.text_proj_in(text_embeddings))

    # Masks
    a_mask = (audio_embeddings.abs().sum(dim=1, keepdim=True) > 0).float()
    t_mask = (text_embeddings.abs().sum(dim=1, keepdim=True) > 0).float()

    # Apply mode ablations
    if mode == "audio":
        t_mask = torch.zeros_like(t_mask)   # zero out text
    elif mode == "text":
        a_mask = torch.zeros_like(a_mask)   # zero out audio

    a = self.norm_audio(a) * a_mask
    t = self.norm_text(t) * t_mask

    # Cross-attention
    a_seq, t_seq = a.unsqueeze(1), t.unsqueeze(1)
    a2t_out, _ = self.cross_attn_audio2text(a_seq, t_seq, t_seq)
    t2a_out, _ = self.cross_attn_text2audio(t_seq, a_seq, a_seq)
    a2t, t2a = a2t_out.squeeze(1), t2a_out.squeeze(1)

    # Gated fusion
    gate_a = self.gate_act(self.gate(torch.cat([a, a2t], dim=1)))
    gate_t = self.gate_act(self.gate(torch.cat([t, t2a], dim=1)))
    fused_a = gate_a * a + (1.0 - gate_a) * a2t
    fused_t = gate_t * t + (1.0 - gate_t) * t2a

    # Fusion attention
    attn_logits = self.attn_score(torch.cat([fused_a, fused_t], dim=1))
    attn_logits = attn_logits + torch.stack([
        self.audio_bias.expand(attn_logits.size(0)),
        torch.zeros_like(attn_logits[:, 1])
    ], dim=1)
    attn_w = F.softmax(attn_logits, dim=1)

    fused = (fused_a * attn_w[:, 0].unsqueeze(1) * a_mask) + \
            (fused_t * attn_w[:, 1].unsqueeze(1) * t_mask)
    fused = self.shared_ln(fused)

    x = self.dropout(self.relu(self.mlp1(fused)))
    x = self.dropout(self.relu(self.mlp2(x)))

    return self.sentiment_head(x), self.emotion_head(x), attn_w
