import os
import argparse
from pathlib import Path
from tqdm import tqdm
import logging
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from sprint_layer.dialogue_xai import DialogueXAI

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Batch Generate Dialogue XAI Reports")
    parser.add_argument("--calls_dir", default="results/calls", help="Directory containing sprint JSON outputs")
    parser.add_argument("--output_dir", default="results/xai_dialogues", help="Output directory for plots")
    parser.add_argument("--report_dir", default="results/xai_reports", help="Output directory for MD reports")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of calls to process")
    
    args = parser.parse_args()
    
    calls_path = Path(args.calls_dir)
    if not calls_path.exists():
        logger.error(f"Calls directory not found: {args.calls_dir}")
        return
        
    json_files = list(calls_path.glob("*.json"))
    if args.limit:
        json_files = json_files[:args.limit]
        
    if not json_files:
        logger.warning(f"No JSON files found in {args.calls_dir}")
        return
        
    logger.info(f"Processing {len(json_files)} calls for XAI...")
    
    xai_engine = DialogueXAI(output_dir=args.output_dir, report_dir=args.report_dir)
    
    success_count = 0
    for jf in tqdm(json_files, desc="Generating XAI"):
        try:
            xai_engine.process_call(jf)
            success_count += 1
        except Exception as e:
            logger.error(f"Failed to process {jf.name}: {e}")
            
    logger.info(f"Finished! Successfully generated reports for {success_count}/{len(json_files)} calls.")
    logger.info(f"Reports saved to: {args.report_dir}")
    logger.info(f"Visualizations saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
