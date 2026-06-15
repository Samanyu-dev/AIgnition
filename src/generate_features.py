import argparse, glob, os, pickle, pandas as pd, numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='./data')
parser.add_argument('--out', default='features.pkl')
args = parser.parse_args()

def find_file(data_dir, keyword):
    patterns = [f'*{keyword}*', f'*{keyword.lower()}*', f'*{keyword.upper()}*']
    for p in patterns:
        matches = glob.glob(os.path.join(data_dir, p))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"No file matching '{keyword}' found in {data_dir}")

bing_path  = find_file(args.data_dir, 'bing')
goog_path  = find_file(args.data_dir, 'google')
meta_path  = find_file(args.data_dir, 'meta')

bing = pd.read_csv(bing_path)
goog = pd.read_csv(goog_path)
meta = pd.read_csv(meta_path)
