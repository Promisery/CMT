import argparse
import yaml
from train_eval.ensembler import Ensembler
from train_eval.utils import seed_everything
import os

if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--configs", help="Config file with dataset parameters", required=True, nargs='+')
    parser.add_argument("-r", "--data_root", help="Root directory with data", required=True)
    parser.add_argument("-d", "--data_dir", help="Directory to extract data", required=True)
    parser.add_argument("-o", "--output_dir", help="Directory to save results", required=True)
    parser.add_argument("-w", "--checkpoints", help="Path to pre-trained or intermediate checkpoint", required=True, nargs='+')
    parser.add_argument("-s", "--seed", type=int, help="Random seed for everything", default=2024)
    args = parser.parse_args()

    seed_everything(args.seed)

    # Make directories
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    if not os.path.isdir(os.path.join(args.output_dir, 'results')):
        os.mkdir(os.path.join(args.output_dir, 'results'))


    # Load config
    cfgs = []
    for config in args.configs:
        with open(config, 'r') as yaml_file:
            cfgs.append(yaml.safe_load(yaml_file))

    # Evaluate
    ensembler = Ensembler(cfgs, args.data_root, args.data_dir, args.checkpoints)
    ensembler.evaluate(output_dir=args.output_dir)
    # evaluator.generate_nuscenes_benchmark_submission(output_dir=args.output_dir)