# Auxiliar function that creates .lst files made by Pedro Aurélio Coelho de ALmeida on the 21st of September, 2021
import argparse
import pathlib
import os
from glob import glob
# usage example:
# python create_lst_files.py -i ../../datasets/MSRA-B/imgs/ -f ../../datasets/MSRA-B/msra_b_valid.txt -o ../../datasets/MSRA-B/msra_b_val.lst -g ../../datasets/MSRA-B/gt/ -n ../../datasets/MSRA-B/pseudo_labels/full_dataset/
def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--img_dir', help='Path to images dir', default=None, type=str)
	parser.add_argument('-f', '--files', help='Txt train, test or validation images that will make up the .lst file', default=None, type=str)
	parser.add_argument('-o', '--out', help='LST output file',default=None, type=str)
	parser.add_argument('-g', '--gt_dir', help='Path to ground truth dir', default=None, type=str, required=False)
	parser.add_argument('-n', '--noisy_dir', help='Path to noisy labels dir', default=None, type=str, required=False)

	return parser.parse_args()

if __name__ == "__main__":
	args = parse_args()
	with open(args.files, 'r') as f:
		files = [pathlib.Path(x).stem for x in f.readlines() if not x.isspace()] # remove any extension should the database train/test/val definition have one

	with open(args.out, 'w') as f:
		for file in files:
			img_file = os.path.basename(glob(args.img_dir + file + ".*")[0])
			if args.gt_dir:
				gt_file = os.path.basename(glob(args.gt_dir + file + ".*")[0])
			else:
				gt_file="NONE"

			if args.noisy_dir:
				noisy_labelers = tuple(["_DSR.png", "_MC.png", "_res.png", "_wCtr_Optimized.png"]) # DSR, MC, HS and RBD labelers respectively
				noisy_files = "\t".join([os.path.basename(x) for x in glob(args.noisy_dir + file + "_*") if x.endswith(noisy_labelers)])
			else:
				noisy_files = ""

			f.write("{}\t{}\t{}\n".format(img_file, gt_file, noisy_files))
