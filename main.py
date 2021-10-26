import argparse

from find_duplicates import CountDuplicates

parser = argparse.ArgumentParser()
parser.add_argument('-dp', '--data_path', default='./',
                    type=str, help='Path to dataset of images')
parser.add_argument('-met', '--metric', type=str,
                    default='abs_diff', help='abs_diff or ahash')
args = parser.parse_args()

counter = CountDuplicates(args.data_path, args.metric)
num_duplicates = counter()

print('Number of duplicates using {} is {}'.format(args.metric, num_duplicates))