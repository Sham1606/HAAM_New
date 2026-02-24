#!/usr/bin/env python
"""
Convenience runner: aggregates sprint features then trains the Marathon LSTM.

Usage:
    python scripts/train_marathon_lstm.py
    python scripts/train_marathon_lstm.py --epochs 80 --no-aggregation
"""
import sys
import os
import argparse
import logging

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train Marathon LSTM Risk Predictor")
    parser.add_argument("--epochs", type=int, default=60, help="Training epochs (default: 60)")
    parser.add_argument("--no-aggregation", action="store_true",
                        help="Skip feature aggregation step (use existing agent_features.csv)")
    args = parser.parse_args()

    # Step 1: Aggregate sprint features (unless skipped)
    if not args.no_aggregation:
        logger.info("Step 1/2: Running Marathon feature aggregation...")
        try:
            from src.marathon_layer.aggregate_features import run_aggregation
            success = run_aggregation(
                input_dir="results/calls",
                output_dir="results/marathon"
            )
            if success:
                logger.info("Aggregation complete.")
            else:
                logger.warning("Aggregation returned no data — will use synthetic data for training.")
        except Exception as e:
            logger.warning(f"Aggregation failed ({e}) — will use synthetic data for training.")
    else:
        logger.info("Skipping aggregation (--no-aggregation flag set).")

    # Step 2: Train LSTM
    logger.info("Step 2/2: Training LSTM Risk Predictor...")
    from src.marathon_layer.train_risk_predictor import train_model
    train_model(epochs=args.epochs)
    logger.info("Done! Model saved to saved_models/marathon_risk_predictor.pth")
    logger.info("The Marathon risk scoring engine will now use the LSTM for hybrid predictions.")


if __name__ == "__main__":
    main()
