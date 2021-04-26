import argparse

import numpy as np
from make_data import generate_data
from trainer import train
from explainer import explain

def create_data(args): 
	"""
	Create train and validation datasets.
	"""
	print("Generating training dataset..")
	X_train, y_train, _ = generate_data(args)  
	
	print("Generating val dataset..")
	X_val, y_val, datatypes_val = generate_data(args)  

	input_shape = X_train.shape[1] 

	return X_train, y_train, X_val, y_val, datatypes_val, input_shape


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.version = '0.1'
	parser.add_argument('-v', action='version')

	parser.add_argument('-d', '--datatype', type = str, 
		choices = ['random', 'sum', 'orange_skin','XOR','nonlinear_additive','switch'], default = 'sum', help='type of dataset')
	parser.add_argument('-n','--n_samples', type = int, default = int(100), help='number of samples to generate')
	parser.add_argument('-f','--n_feats', type = int, default = 3, help='number of feature dimension to generate')
	# parser.add_argument('--train', action='store_true',help='Operating in training mode?')

	args = parser.parse_args()

	data = create_data(args)
	model = train(data, load_existing = False)
	explain(model, data, args)

	print("Script Complete.")