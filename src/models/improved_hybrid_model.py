import torch
import torch.nn as nn
import torch.nn.functional as F

class ImprovedHybridModel(nn.Module):
    """
    Improved Hybrid Fusion Model for Emotion Recognition.
    
    Inputs:
    - Acoustic: 12 features (Pitch, Jitter, Shimmer, HNR, RMS, ZCR, Rate, Spectral)
    - Text: 5 emotion probabilities + 768-dim RoBERTa embedding
    
    Architecture:
    - Acoustic Branch (MLP)
    - Text Branch (MLP)
    - Late Fusion via Concatenation
    - Classification Head
    """
    
    def __init__(self, n_acoustic=12, n_text_emb=768, n_text_probs=5, n_classes=5, dropout=0.3):
        super(ImprovedHybridModel, self).__init__()
        
        # === Acoustic Branch ===
        # Input: 12
        self.acoustic_net = nn.Sequential(
            nn.Linear(n_acoustic, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        
        # === Text Branch ===
        # Input: 768 (embedding) + 5 (probabilities) = 773
        self.text_input_dim = n_text_emb + n_text_probs
        self.text_net = nn.Sequential(
            nn.Linear(self.text_input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        
        # === Fusion Head ===
        # 32 (Acoustic) + 128 (Text) = 160
        self.fusion_dim = 32 + 128
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(64, n_classes)
        )
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x_acoustic, x_text_emb, x_text_probs):
        """
        Forward pass.
        
        Args:
            x_acoustic: [Batch, 12]
            x_text_emb: [Batch, 768] (RoBERTa CLS)
            x_text_probs: [Batch, 5] (Emotion probabilities)
        """
        # Checks
        if x_text_emb.dim() == 1: x_text_emb = x_text_emb.unsqueeze(0)
        
        # 1. Acoustic Process
        a_out = self.acoustic_net(x_acoustic)
        
        # 2. Text Process
        # Concatenate embedding and probabilities
        t_input = torch.cat([x_text_emb, x_text_probs], dim=1)
        t_out = self.text_net(t_input)
        
        # 3. Fusion
        combined = torch.cat([a_out, t_out], dim=1)
        
        # 4. Classification
        logits = self.classifier(combined)
        
        return logits

    def predict(self, x_acoustic, x_text_emb, x_text_probs):
        """Inference helper."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x_acoustic, x_text_emb, x_text_probs)
            probs = F.softmax(logits, dim=1)
        return probs

if __name__ == "__main__":
    # Test Architecture
    model = ImprovedHybridModel()
    print("Model Architecture:")
    print(model)
    
    # Dummy Input
    bs = 4
    x_ac = torch.randn(bs, 12)
    x_temb = torch.randn(bs, 768)
    x_tprob = torch.rand(bs, 5)
    
    out = model(x_ac, x_temb, x_tprob)
    print(f"\nOutput Shape: {out.shape}")
