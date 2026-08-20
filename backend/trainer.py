import argparse
import os
import sys
import logging
from model_utils import train_and_save_model

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('training.log', mode='w')
        ]
    )

    parser = argparse.ArgumentParser(description="Train Harmonium Note Recognizer ML Model")
    parser.add_argument('--dataset', default='notes_dataset', help='Path to dataset directory')
    parser.add_argument('--out', default='model.pkl', help='Output model path (.pkl)')
    parser.add_argument('--sr', type=int, default=22050, help='Audio sample rate for feature extraction')
    parser.add_argument('--test-size', type=float, default=0.15, help='Validation test split ratio')
    parser.add_argument('--max-samples', type=int, default=None, help='Optional max number of samples (for fast test training)')
    args = parser.parse_args()

    # Search candidates for dataset path
    candidates = [
        args.dataset,
        os.path.join(os.path.dirname(__file__), args.dataset),
        os.path.join(os.path.dirname(__file__), '..', args.dataset),
        os.path.join('..', args.dataset)
    ]
    
    dataset_dir = None
    for c in candidates:
        abs_c = os.path.abspath(c)
        if os.path.isdir(abs_c):
            dataset_dir = abs_c
            break

    if not dataset_dir:
        logging.error(f"Dataset directory not found in candidates: {[os.path.abspath(c) for c in candidates]}")
        sys.exit(1)

    logging.info(f"Starting training pipeline...")
    logging.info(f"Dataset path: {dataset_dir}")
    logging.info(f"Output path: {args.out}")
    logging.info(f"Sample rate: {args.sr}")

    try:
        X, y = train_and_save_model(
            dataset_dir=dataset_dir,
            out_path=args.out,
            sr=args.sr,
            test_size=args.test_size,
            max_samples=args.max_samples
        )
        logging.info("Training pipeline completed successfully.")
    except Exception as e:
        logging.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)
