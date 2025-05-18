import argparse
import yaml
from train_eval.evaluator import Evaluator
from train_eval.utils import seed_everything
import os

if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Config file with dataset parameters", required=True)
    parser.add_argument("-r", "--data_root", help="Root directory with data", required=True)
    parser.add_argument("-d", "--data_dir", help="Directory to extract data", required=True)
    parser.add_argument("-o", "--output_dir", help="Directory to save results", required=True)
    parser.add_argument("-w", "--checkpoint", help="Path to pre-trained or intermediate checkpoint", required=True)
    parser.add_argument("-s", "--seed", type=int, help="Random seed for everything", default=2024)
    args = parser.parse_args()

    seed_everything(args.seed)

    # Make directories
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    if not os.path.isdir(os.path.join(args.output_dir, 'results')):
        os.mkdir(os.path.join(args.output_dir, 'results'))


    # Load config
    with open(args.config, 'r') as yaml_file:
        cfg = yaml.safe_load(yaml_file)


    # Evaluate
    evaluator = Evaluator(cfg, args.data_root, args.data_dir, args.checkpoint)
    evaluator.evaluate(output_dir=args.output_dir)
    # evaluator.generate_nuscenes_benchmark_submission(output_dir=args.output_dir)