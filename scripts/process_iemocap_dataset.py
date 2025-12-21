
import os
import glob
import json
import logging
import argparse
import sys
import time
import re
from tqdm import tqdm
from datetime import datetime

# Add project root to path to import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.sprint_layer.run_sprint_pipeline import SprintPipeline
except ImportError:
    # Fallback if run from scripts dir
    sys.path.append(os.path.join(os.getcwd(), 'src'))
    from sprint_layer.run_sprint_pipeline import SprintPipeline

# Configuration
IEMOCAP_ROOT = r"D:\haam_framework\iemocapfullrelease\1\IEMOCAP_full_release"
OUTPUT_DIR = r"D:\haam_framework\results\calls_iemocap"
LOG_FILE = "processing_errors.log"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Emotion Mapping (IEMOCAP 11 -> HAAM 6)
EMOTION_MAP = {
    'neu': 'neutral',
    'hap': 'joy',
    'sad': 'sadness',
    'ang': 'anger',
    'exc': 'joy',      # Excitement maps to Joy
    'fru': 'anger',    # Frustration maps to Anger
    'fea': 'fear',
    'sur': 'neutral',  # Surprise is ambiguous, mapping to neutral
    'dis': 'disgust',
    'oth': 'neutral',  # Other
    'xxx': 'neutral'   # Undecided
}

def parse_iemocap_label(label_file):
    """
    Parses IEMOCAP EmoEvaluation file.
    Returns dict: {utterance_id: {'emotion': mapped_emo, 'raw': raw_emo, 'vals': [v, a, d]}}
    """
    labels = {}
    if not os.path.exists(label_file):
        logger.warning(f"Label file not found: {label_file}")
        return labels

    with open(label_file, 'r') as f:
        content = f.read()
    
    # Regex to extract categorical labels
    # Format: [START - END]  TURN_NAME  EMOTION  [V, A, D]
    # Updated to handle uppercase in ID and flexible whitespace
    pattern = re.compile(r"\[(\d+\.\d+)\s*-\s*(\d+\.\d+)\]\s+(Ses0\d[MF]_[A-Za-z0-9_]+)\s+([a-z]{3})\s+\[(\d+\.\d+),\s*(\d+\.\d+),\s*(\d+\.\d+)\]")
    
    matches = pattern.findall(content)
    for match in matches:
        start, end, utt_id, raw_emo, v, a, d = match
        labels[utt_id] = {
            'emotion': EMOTION_MAP.get(raw_emo, 'neutral'),
            'raw_emotion': raw_emo,
            'valence': float(v),
            'arousal': float(a),
            'dominance': float(d),
            'start': float(start),
            'end': float(end)
        }
    return labels

def get_transcript(transcript_dir, dialogue_id):
    """Retrieves transcript text for a dialogue."""
    txt_path = os.path.join(transcript_dir, f"{dialogue_id}.txt")
    transcripts = {}
    if os.path.exists(txt_path):
        with open(txt_path, 'r') as f:
            for line in f:
                # Format: Ses01F_impro01_F000 [start-end]: text
                parts = line.strip().split(']: ')
                if len(parts) == 2:
                    uid = parts[0].split(' [')[0]
                    text = parts[1]
                    transcripts[uid] = text
    return transcripts

def process_session(session_id, pipeline):
    """Process all files in a session."""
    session_dir = os.path.join(IEMOCAP_ROOT, f"Session{session_id}")
    wav_root = os.path.join(session_dir, "sentences", "wav")
    label_root = os.path.join(session_dir, "dialog", "EmoEvaluation")
    trans_root = os.path.join(session_dir, "dialog", "transcriptions")
    
    if not os.path.exists(wav_root):
        logger.error(f"Session {session_id} not found at {wav_root}")
        return

    # Gather all dialogues
    dialogues = [d for d in os.listdir(wav_root) if os.path.isdir(os.path.join(wav_root, d))]
    
    logger.info(f"Processing Session {session_id}: {len(dialogues)} dialogues found.")
    
    stats = {'processed': 0, 'skipped': 0, 'failed': 0}
    
    for dialogue in tqdm(dialogues, desc=f"Session {session_id}"):
        # Load labels for this dialogue (labels are per session or per dialogue? usually per dialogue in Categorical folder but check structure)
        # Structure check: Session1/dialog/EmoEvaluation/Ses01F_impro01.txt usually contains VAD, 
        # but Categorical labels might be in a summary file or individual files.
        # Verified IEMOCAP structure: Session1/dialog/EmoEvaluation/Ses01F_impro01.txt contains the labels.
        
        label_file = os.path.join(label_root, f"{dialogue}.txt")
        labels = parse_iemocap_label(label_file)
        
        transcripts = get_transcript(trans_root, dialogue)
        
        wav_dir = os.path.join(wav_root, dialogue)
        wav_files = glob.glob(os.path.join(wav_dir, "*.wav"))
        
        for wav_path in wav_files:
            utterance_id = os.path.splitext(os.path.basename(wav_path))[0]
            
            # Construct output path
            # call_id format: iemocap_Ses01_Ses01F_impro01_F000
            call_id = f"iemocap_{utterance_id}"
            output_file = os.path.join(OUTPUT_DIR, f"{call_id}.json")
            
            if os.path.exists(output_file):
                stats['skipped'] += 1
                continue
                
            try:
                # Generate Agent ID
                # Ses01F_impro01_F000 -> Agent is F (Female) from Ses01
                # Format: iemocap_agent_01_F
                gender = 'F' if '_F' in utterance_id.split('_')[-1] else 'M'
                agent_id = f"iemocap_agent_{session_id:02d}_{gender}"
                
                # Check ground truth
                if utterance_id not in labels:
                    # Skip files without labels (sometimes metadata files exist)
                    continue
                    
                gt = labels[utterance_id]
                transcript_text = transcripts.get(utterance_id, "")
                
                # Process Call
                # Note: We pass the text if we want to skip Whisper, but pipeline does whisper.
                # Ideally, we let pipeline do Whisper to be consistent with CREMA-D processing.
                # But we save GT transcript in metadata.
                
                result = pipeline.process_call(wav_path, agent_id, call_id)
                
                # Enrich result with IEMOCAP metadata
                result['ground_truth'] = {
                    'emotion': gt['emotion'],
                    'raw_emotion': gt['raw_emotion'],
                    'valence': gt['valence'],
                    'arousal': gt['arousal'],
                    'dominance': gt['dominance']
                }
                
                result['metadata'] = {
                    'dataset': 'IEMOCAP',
                    'session': session_id,
                    'dialogue': dialogue,
                    'utterance_id': utterance_id,
                    'transcript_ground_truth': transcript_text
                }
                
                # Save
                with open(output_file, 'w') as f:
                    json.dump(result, f, indent=2)
                
                stats['processed'] += 1
                
            except Exception as e:
                logger.error(f"Failed to process {utterance_id}: {e}")
                stats['failed'] += 1
                
    logger.info(f"Session {session_id} Complete. Stats: {stats}")

def main():
    print("==================================================")
    print("   IEMOCAP DATASET PROCESSING UTILITY")
    print("==================================================")
    
    if not os.path.exists(IEMOCAP_ROOT):
        print(f"Error: IEMOCAP root not found at {IEMOCAP_ROOT}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Initialize Pipeline
    print("Initializing Sprint Pipeline...")
    try:
        pipeline = SprintPipeline()
    except Exception as e:
        print(f"Failed to init pipeline: {e}")
        return

    # Process all 5 sessions
    for i in range(1, 6):
        process_session(i, pipeline)
        
    print("\nProcessing Complete.")

if __name__ == "__main__":
    main()
