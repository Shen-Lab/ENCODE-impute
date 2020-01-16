import argparse
import numpy as np
from random import random

def parse_arguments():
	parser = argparse.ArgumentParser(prog = 'chromsome id', description = '')
	parser.add_argument('chr_id', type = str)
	args = parser.parse_args()
	return args

def onehotencode(notation):
	A = np.array([[1, 0, 0, 0]], dtype = bool)
	G = np.array([[0, 1, 0, 0]], dtype = bool)
	C = np.array([[0, 0, 1, 0]], dtype = bool)
	T = np.array([[0, 0, 0, 1]], dtype = bool)
	random_temp = random()
	if(notation == 'A'):
		output_temp = A
	elif(notation == 'G'):
		output_temp = G
	elif(notation == 'C'):
		output_temp = C
	elif(notation == 'T'):
		output_temp = T
	elif(notation == 'W'):
		if(random_temp < 0.5):
			output_temp = A
		else:
			output_temp = T
	elif(notation == 'S'):
		if(random_temp < 0.5):
			output_temp = C
		else:
			output_temp = G
	elif(notation == 'M'):
		if(random_temp < 0.5):
			output_temp = A
		else:
			output_temp = C
	elif(notation == 'K'):
		if(random_temp < 0.5):
			output_temp = G
		else:
			output_temp = T
	elif(notation == 'R'):
		if(random_temp < 0.5):
			output_temp = A
		else:
			output_temp = G
	elif(notation == 'Y'):
		if(random_temp < 0.5):
			output_temp = C
		else:
			output_temp = T
	elif(notation == 'B'):
		if(random_temp < 0.33):
			output_temp = C
		elif(random_temp < 0.67):
			output_temp = G
		else:
			output_temp = T
	elif(notation == 'D'):
		if(random_temp < 0.33):
			output_temp = A
		elif(random_temp < 0.67):
			output_temp = G
		else:
			output_temp = T
	elif(notation == 'H'):
		if(random_temp < 0.33):
			output_temp = A
		elif(random_temp < 0.67):
			output_temp = C
		else:
			 output_temp = T
	elif(notation == 'V'):
		if(random_temp < 0.33):
			output_temp = A
		elif(random_temp < 0.67):
			output_temp = C
		else:
			output_temp = G
	elif(notation == 'N'):
		if(random_temp < 0.25):
			output_temp = A
		elif(random_temp < 0.5):
			output_temp = C
		elif(random_temp < 0.75):
			output_temp = G
		else:
			output_temp = T
	else:
		print('not found')
	return(output_temp)

def main():
	args = parse_arguments()
	chr_id = args.chr_id
	input = open(chr_id + '.fa').readlines()[0].upper()
	output = np.array(np.zeros([len(input)-1, 4]), dtype = bool)
	for i in range(int(len(input)-1)):
		output[i, :] = onehotencode(input[i])
	np.save(chr_id, output)

if __name__=='__main__':
	main()

