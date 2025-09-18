import torch
from torch.utils.data import Dataset, DataLoader
import json
import os
import pandas as pd
import numpy as np

class MELDMultimodalDataset(Dataset):
    """
    Custom PyTorch Dataset for loading the preprocessed MELD data.
    This class reads the JSON files created by `process_meld.py` and prepares
    the audio features, text embeddings, and labels for the model.
    """
    def __init__(self, project_root, data_type='train'):
        """
        Args:
            project_root (str): The absolute path to the project's root directory.
            data_type (str): 'train', 'dev', or 'test' to specify which dataset to load.
        """
        self.processed_dir = os.path.join(project_root, 'data', 'processed', 'json')
        raw_data_dir = os.path.join(project_root, 'data', 'raw')

        # --- ROBUST FILE LIST GENERATION ---
        # Instead of guessing file ranges, we now read the official CSVs to get the correct file lists.
        if data_type == 'train':
            csv_path = os.path.join(raw_data_dir, 'train_sent_emo.csv')
        elif data_type == 'dev':
            csv_path = os.path.join(raw_data_dir, 'dev_sent_emo.csv')
        else: # test
            csv_path = os.path.join(raw_data_dir, 'test_sent_emo.csv')

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"MELD label file not found at: {csv_path}")

        df = pd.read_csv(csv_path)
        self.file_list = [f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.json" for _, row in df.iterrows()]
        
        # Filter the list to only include files that were successfully processed
        self.file_list = [f for f in self.file_list if os.path.exists(os.path.join(self.processed_dir, f))]

        print(f"Found {len(self.file_list)} processed files for the '{data_type}' set.")
        
        # Define mappings from string labels to integer indices
        self.sentiment_mapping = {'positive': 0, 'negative': 1, 'neutral': 2}
        self.emotion_mapping = {
            'neutral': 0, 'joy': 1, 'sadness': 2, 'fear': 3, 
            'anger': 4, 'surprise': 5, 'disgust': 6
        }

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.file_list)

    def __getitem__(self, idx):
        """
        Fetches one sample from the dataset at the given index.
        """
        file_path = os.path.join(self.processed_dir, self.file_list[idx])
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            print(f"Warning: Could not load or decode JSON file: {file_path}. Skipping.")
            # Return a dummy sample or the next valid one
            return self.__getitem__((idx + 1) % len(self))

        # --- Extract Features ---
        audio_features = data.get('audio_features', {})
        mfccs = audio_features.get('mfccs', [0.0]*13)
        if len(mfccs) > 13: mfccs = mfccs[:13]
        if len(mfccs) < 13: mfccs += [0.0] * (13 - len(mfccs))
        
        full_audio_vec = mfccs + [audio_features.get('pitch', 0.0), audio_features.get('energy', 0.0)]
        
        text_embeddings = data.get('text_embeddings', [0.0]*768)
        if not text_embeddings or len(text_embeddings) != 768: 
            text_embeddings = [0.0]*768

        # --- Extract Labels ---
        sentiment_label = self.sentiment_mapping.get(data.get('sentiment'), 2) # Default to neutral
        emotion_label = self.emotion_mapping.get(data.get('emotion'), 0) # Default to neutral

        # --- Convert to Tensors ---
        sample = {
            'audio_features': torch.tensor(full_audio_vec, dtype=torch.float32),
            'text_embeddings': torch.tensor(text_embeddings, dtype=torch.float32),
            'sentiment_label': torch.tensor(sentiment_label, dtype=torch.long),
            'emotion_label': torch.tensor(emotion_label, dtype=torch.long)
        }
        
        return sample

# --- EXAMPLE USAGE ---
if __name__ == '__main__':
    # Automatically detect the project root directory
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
    except NameError:
        project_root = os.getcwd() # Fallback for interactive environments

    print(f"Project Root Detected: {project_root}")
    
    print("\nTesting the training dataset loader...")
    # Pass the project root to the dataset class
    train_dataset = MELDMultimodalDataset(project_root=project_root, data_type='train')
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    try:
        first_batch = next(iter(train_loader))
        
        print("\n--- First Batch Inspection ---")
        print(f"Batch size: {len(first_batch['sentiment_label'])}")
        print(f"Audio features shape: {first_batch['audio_features'].shape}")
        print(f"Text embeddings shape: {first_batch['text_embeddings'].shape}")
        print(f"Sentiment labels shape: {first_batch['sentiment_label'].shape}")
        print(f"Emotion labels shape: {first_batch['emotion_label'].shape}")
        print("\nData loader is working correctly!")
        
    except StopIteration:
        print("\nCould not load a batch. Is your 'processed/json' directory empty or are there too few files for the 'train' split?")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
