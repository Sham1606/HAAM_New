import torch
import os
import sys
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
# add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_processing.embedding_dataset import MELDEmbeddingDataset
from models.fusion_model import MultimodalFusionModel

def load_model(model_path, device):
    model = MultimodalFusionModel()
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

def predict_and_analyze(model, dataloader, device):
    all_sent_preds, all_emo_preds = [], []
    all_sent_trues, all_emo_trues = [], []
    all_dialogue, all_utt = [], []
    all_audio_attn = []

    with torch.no_grad():
        for batch in dataloader:
            audio = batch['audio_embedding'].to(device)
            text = batch['text_embedding'].to(device)
            s_true = batch['sentiment_label'].numpy()
            e_true = batch['emotion_label'].numpy()

            s_logits, e_logits, attn_w = model(audio, text)
            s_pred = torch.argmax(s_logits, dim=1).cpu().numpy()
            e_pred = torch.argmax(e_logits, dim=1).cpu().numpy()
            audio_attn = attn_w[:,0].cpu().numpy()  # audio weight

            all_sent_preds.extend(s_pred.tolist())
            all_emo_preds.extend(e_pred.tolist())
            all_sent_trues.extend(s_true.tolist())
            all_emo_trues.extend(e_true.tolist())
            all_dialogue.extend(batch.get('dialogue_id', [-1]*len(s_pred)))
            all_utt.extend(batch.get('utterance_id', [-1]*len(s_pred)))
            all_audio_attn.extend(audio_attn.tolist())

    return {
        "sent_pred": np.array(all_sent_preds),
        "emo_pred": np.array(all_emo_preds),
        "sent_true": np.array(all_sent_trues),
        "emo_true": np.array(all_emo_trues),
        "dialogue": np.array(all_dialogue),
        "utterance": np.array(all_utt),
        "audio_attn": np.array(all_audio_attn)
    }

if __name__ == "__main__":
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        PROJECT_ROOT = os.path.dirname(os.path.dirname(script_dir))
    except NameError:
        PROJECT_ROOT = os.getcwd()

    MODEL_PATH = os.path.join(PROJECT_ROOT, 'saved_models', 'sprint_model_v5_best.pth')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load test dataset
    test_dataset = MELDEmbeddingDataset(project_root=PROJECT_ROOT, data_type='test')
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    print(f"Loaded {len(test_dataset)} test samples.")

    # Load model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Train first.")
    model = load_model(MODEL_PATH, device)
    print("Model loaded.")

    # Predict
    results = predict_and_analyze(model, test_loader, device)

    # Map ids back to labels
    sent_id2label = {v:k for k,v in test_dataset.sentiment_mapping.items()}
    emo_id2label = {v:k for k,v in test_dataset.emotion_mapping.items()}

    sent_pred_labels = [sent_id2label[int(i)] for i in results['sent_pred']]
    emo_pred_labels = [emo_id2label[int(i)] for i in results['emo_pred']]
    sent_true_labels = [sent_id2label[int(i)] for i in results['sent_true']]
    emo_true_labels = [emo_id2label[int(i)] for i in results['emo_true']]

    # Overall metrics
    sent_acc = accuracy_score(results['sent_true'], results['sent_pred'])
    emo_acc = accuracy_score(results['emo_true'], results['emo_pred'])
    sent_f1 = f1_score(results['sent_true'], results['sent_pred'], average='macro', zero_division=0)
    emo_f1 = f1_score(results['emo_true'], results['emo_pred'], average='macro', zero_division=0)
    print(f"Sentiment: Acc={sent_acc:.4f}, F1={sent_f1:.4f}")
    print(f"Emotion:   Acc={emo_acc:.4f}, F1={emo_f1:.4f}")

    # Per-class average audio attention (group by true emotion)
    df = pd.DataFrame({
        "Dialogue_ID": results['dialogue'],
        "Utterance_ID": results['utterance'],
        "true_emotion": results['emo_true'],
        "pred_emotion": results['emo_pred'],
        "audio_attn": results['audio_attn']
    })

    per_class_attn = df.groupby('true_emotion')['audio_attn'].agg(['mean','count']).reset_index()
    per_class_attn['emotion_label'] = per_class_attn['true_emotion'].map(emo_id2label)
    per_class_attn = per_class_attn[['true_emotion','emotion_label','mean','count']].rename(columns={'mean':'avg_audio_attn','count':'n_samples'})
    print("\nPer-emotion average audio attention:")
    print(per_class_attn.to_string(index=False))

    # Save predictions CSV (with attention)
    out_df = pd.DataFrame({
        "Dialogue_ID": results['dialogue'],
        "Utterance_ID": results['utterance'],
        "True_Emotion": [emo_id2label[int(i)] for i in results['emo_true']],
        "Pred_Emotion": [emo_id2label[int(i)] for i in results['emo_pred']],
        "True_Sentiment": [sent_id2label[int(i)] for i in results['sent_true']],
        "Pred_Sentiment": [sent_id2label[int(i)] for i in results['sent_pred']],
        "Audio_Attention": results['audio_attn']
    })
    out_path = os.path.join(PROJECT_ROOT, 'predictions_with_attention.csv')
    out_df.to_csv(out_path, index=False)
    print(f"\nPredictions with attention saved to {out_path}")
