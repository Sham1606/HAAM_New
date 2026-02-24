import pickle
import sys
import os

model_path = r"d:\haam\HAAM_New\models\acoustic_emotion_model.pkl"

try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print(f"Model type: {type(model)}")
    print(f"Model attributes: {dir(model)}")
except Exception as e:
    print(f"Error loading model: {e}")
