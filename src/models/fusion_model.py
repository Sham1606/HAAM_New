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

    def forward(self, audio_embeddings, text_embeddings):
        """
        Inputs:
            audio_embeddings: (B, audio_dim)
            text_embeddings:  (B, text_dim)
        Returns:
            sentiment_logits, emotion_logits, attn_w (B,2)
        """
        # Project
        a = self.relu(self.audio_proj_in(audio_embeddings))  # (B, H)
        t = self.relu(self.text_proj_in(text_embeddings))    # (B, H)

        # Masks: 1 for valid, 0 for placeholders
        a_mask = (audio_embeddings.abs().sum(dim=1, keepdim=True) > 0).float()
        t_mask = (text_embeddings.abs().sum(dim=1, keepdim=True) > 0).float()

        # Norm + apply masks
        a = self.norm_audio(a) * a_mask
        t = self.norm_text(t) * t_mask

        # Convert to seq len=1 for MultiheadAttention (batch_first=True)
        a_seq = a.unsqueeze(1)  # (B,1,H)
        t_seq = t.unsqueeze(1)  # (B,1,H)

        # Cross-attention: audio queries text
        # If text is missing t_seq is zeroed out above
        a2t_out, a2t_w = self.cross_attn_audio2text(query=a_seq, key=t_seq, value=t_seq, need_weights=True)
        a2t = a2t_out.squeeze(1)  # (B,H)

        # Cross-attention: text queries audio
        t2a_out, t2a_w = self.cross_attn_text2audio(query=t_seq, key=a_seq, value=a_seq, need_weights=True)
        t2a = t2a_out.squeeze(1)  # (B,H)

        # Gates (GMU-style) to combine original + cross outputs
        concat_a = torch.cat([a, a2t], dim=1)      # (B, 2H)
        gate_a = self.gate_act(self.gate(concat_a)) # (B, H)

        concat_t = torch.cat([t, t2a], dim=1)
        gate_t = self.gate_act(self.gate(concat_t))

        fused_a = gate_a * a + (1.0 - gate_a) * a2t
        fused_t = gate_t * t + (1.0 - gate_t) * t2a

        # Fusion attention logits (audio/text)
        combined = torch.cat([fused_a, fused_t], dim=1)  # (B, 2H)
        attn_logits = self.attn_score(combined)          # (B,2)
        # Add small bias to audio logit to encourage audio if useful
        attn_logits = attn_logits + torch.stack([self.audio_bias.expand(attn_logits.size(0)), torch.zeros_like(attn_logits[:,1])], dim=1)
        attn_w = F.softmax(attn_logits, dim=1)          # (B,2)

        # Weighted fusion with masks
        fused = (fused_a * attn_w[:, 0].unsqueeze(1) * a_mask) + (fused_t * attn_w[:, 1].unsqueeze(1) * t_mask)
        fused = self.shared_ln(fused)

        # Shared MLP
        x = self.dropout(self.relu(self.mlp1(fused)))
        x = self.dropout(self.relu(self.mlp2(x)))

        sentiment_logits = self.sentiment_head(x)
        emotion_logits = self.emotion_head(x)

        return sentiment_logits, emotion_logits, attn_w
