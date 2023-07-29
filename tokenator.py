# Initial script from u/funkmasterplex
# https://www.reddit.com/r/StableDiffusion/comments/154xnmm/comment/jss3mt7/

# Setup
# python3 -m venv venv
# source venv/bin/activate
# pip install -r requirements.txt

# Run
# python3 tokenator.py --file1 <...> --file2 <...>

import torch
import tqdm
import argparse
from safetensors import safe_open
from clip.simple_tokenizer import SimpleTokenizer

def safetensors_load(ckpt, map_location="cpu"):
	tensors = {}
	with safe_open(ckpt, framework="pt", device=map_location) as f:
		for key in f.keys():
			tensors[key] = f.get_tensor(key)
	return tensors

def parse_args():
	parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
	parser.add_argument(
		"--file1",
		type=str,
		help="path to safetensors of model",
	)
	parser.add_argument(
		"--file2",
		type=str,
		help="path to safetensors of model",
	)
	opt = parser.parse_args()
	return opt

opt = parse_args()

tokenizer = SimpleTokenizer()

standard = safetensors_load(opt.file1, map_location='cpu')
if 'state_dict' in standard:
	standard = standard['state_dict']

custom = safetensors_load(opt.file2, map_location='cpu')
if 'state_dict' in custom:
	custom = custom['state_dict']

report = []

error_epsilon = 0.0001

key = 'cond_stage_model.transformer.text_model.embeddings.token_embedding.weight'
temp = []
for x in tqdm.tqdm(range(49408)):
	this_e = torch.sum(torch.abs(custom[key][x].clone().half() - standard[key][x].clone().half()))
	if this_e >= error_epsilon:
		token = tokenizer.decode([x])
		# temp.append([f"\tToken {x} ({token}) error: {this_e}", this_e])
		temp.append([f"{token:12} error: {this_e}", this_e])
temp.sort(key=lambda x: -x[1])
temp = list(map(lambda x: x[0], temp))[:100]
report += temp
report.append('')

print("\n".join(report))
