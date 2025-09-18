# import os
# import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Add src to path
# from data_processing.embedding_dataset import MELDEmbeddingDataset

# script_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.dirname(os.path.dirname(script_dir))
# dataset = MELDEmbeddingDataset(project_root=project_root, data_type='train')

# print(f"Total samples: {len(dataset)}")
# sample = dataset[0]
# print(f"Sample 0: Audio shape {sample['audio_embedding'].shape}, Text shape {sample['text_embedding'].shape}")

# import pandas as pd
# df = pd.read_csv(r'D:\My_Data_Science\fpo\haam_framework\data\raw\train_sent_emo.csv')
# print(len(df))  # Should be ~10,000

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Add src to path
from data_processing.embedding_dataset import MELDEmbeddingDataset

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
dataset = MELDEmbeddingDataset(project_root=project_root, data_type='train')

print(f"Total samples: {len(dataset)}")
sample = dataset[0]
print(f"Sample 0: Audio shape {sample['audio_embedding'].shape}, Text shape {sample['text_embedding'].shape}, Sentiment: {sample['sentiment_label'].item()}, Emotion: {sample['emotion_label'].item()}")

# from moviepy.video.io.VideoFileClip import VideoFileClip
# clip = VideoFileClip(r"D:\My_Data_Science\fpo\haam_framework\data\raw\MELD.Raw\train\dia1038_utt17.mp4")  # Replace with a sample MP4
# clip.audio.write_audiofile("D:\My_Data_Science\fpo\haam_framework\src\training\test.wav", logger=None)