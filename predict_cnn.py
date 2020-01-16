import argparse
import numpy as np
import tensorflow as tf
import time

path = '/scratch/user/wudi930320/imputation_challenge/'

seq_path = path + 'data/chr_npy/'
model_path = path + 'model/cnn_model/'
output_path = path + 'prediction/'

chromosome_index = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 'chr20', 'chr21', 'chr22', 'chrX']
chromosome_index_size = np.array((248956422, 242193529, 198295559, 190214555, 181538259, 170805979, 159345973, 145138636, 138394717, 133797422, 135086622, 133275309, 114364328, 107043718, 101991189, 90338345, 83257441, 80373285, 58617616, 64444167, 46709983, 50818468, 156040895))


cell_embedding_size = 20
assay_embedding_size = 12

batch_size = 256

def parse_arguments():
        parser = argparse.ArgumentParser(prog = 'Calculate the prediction results for each 25bp in a chromosome', description = '')
        parser.add_argument('chr_input', type = str)
        parser.add_argument('assay_cell', type = str)
        args = parser.parse_args()
        return args


def seq_represent(seq):
	x = tf.reshape(seq, [-1, 4, 1000, 1])
	x = tf.layers.conv2d(inputs = x, filters = 320, kernel_size = [1, 8], padding = 'valid', activation = tf.nn.relu)
	x = tf.layers.max_pooling2d(inputs = x, pool_size = [1,4], strides = 4)
	x = tf.layers.dropout(inputs = x, rate = 0.2)
	x = tf.layers.conv2d(inputs = x, filters = 480, kernel_size = [1,8], padding = 'valid', activation = tf.nn.relu)
	x = tf.layers.max_pooling2d(inputs = x, pool_size = [1,4], strides = 4)
	x = tf.layers.dropout(inputs = x, rate = 0.2)
	x = tf.layers.conv2d(inputs = x, filters = 960, kernel_size = [1, 8], padding = 'valid', activation = tf.nn.relu)
	x = tf.layers.dropout(inputs = x, rate = 0.5)
	x = tf.reshape(x, [-1, 50880])
	x = tf.layers.dense(x, units = 1000)
	x = tf.nn.leaky_relu(x)
	x = tf.layers.dropout(inputs = x, rate = 0.2)
	return tf.layers.dense(x, units = 200)

def cell_embed(cell, embedding_size):
	cell_size = 51
	embedding_size = embedding_size
	cell_embeddings = tf.get_variable('cell_embeddings', [cell_size, embedding_size])
	embedded_cell_ids = tf.nn.embedding_lookup(cell_embeddings, cell)
	return embedded_cell_ids

def assay_embed(assay, embedding_size):
        assay_size = 35
        embedding_size = embedding_size
        assay_embeddings = tf.get_variable('assay_embeddings', [assay_size, embedding_size])
        embedded_assay_ids = tf.nn.embedding_lookup(assay_embeddings, assay)
        return embedded_assay_ids

def dilate_final(x) :
	x = tf.layers.dense(x, units = 200)
	x = tf.nn.leaky_relu(x)
	x = tf.layers.dropout(x, rate = 0.2)
	x = tf.layers.dense(x, units = 100)
	x = tf.nn.leaky_relu(x)
	x = tf.layers.dropout(x, rate = 0.2)
	x = tf.layers.dense(x, units = 50)
	x = tf.nn.leaky_relu(x)
	x = tf.layers.dropout(x, rate = 0.2)
	x = tf.layers.dense(x, units = 30)
	x = tf.nn.leaky_relu(x)
	x = tf.layers.dense(x, units = 15)
	x = tf.layers.dense(x, units = 5)
	x = tf.layers.dense(x, units = 1)
	return tf.reshape(x, [-1, 1])


sequence = tf.placeholder(tf.float32, shape=[None, 4, 1000])
label =tf.placeholder(tf.float64, shape=[None, 1])
cell = tf.placeholder(tf.int32, shape=[None])
assay = tf.placeholder(tf.int32, shape=[None])
training_flag = tf.placeholder(tf.bool)
learning_rate = tf.placeholder(tf.float32, name = 'learning_rate')

seq_pred = seq_represent(sequence)
cell_embedded_result = cell_embed(cell, cell_embedding_size)
assay_embedded_result = assay_embed(assay, assay_embedding_size)
whole_input = tf.concat([seq_pred, cell_embedded_result, assay_embedded_result], -1)
pred = dilate_final(whole_input)
loss_mse = tf.losses.mean_squared_error(labels=label, predictions=pred)
loss_op = tf.reduce_mean(loss_mse)

init = tf.global_variables_initializer()
saver = tf.train.Saver()


def main():
	start = time.time()
	args = parse_arguments()
	chr_index_i = int(args.chr_input)
	cell_id = np.repeat(int(args.assay_cell[1:3]) - 1, batch_size)
	assay_id = np.repeat(int(args.assay_cell[4:6]) - 1, batch_size)
	batch_x = np.zeros([batch_size, 4, 1000])
	with tf.Session() as sess:
		saver.restore(sess, model_path + 'frequency_cnn_0.0001_99.ckpt')
		output = np.zeros([chromosome_index_size[chr_index_i]//25 + 1]) + 0.25
		training_seq_input = np.load(seq_path + chromosome_index[chr_index_i] + '.npy')
		for bp_index_i in np.arange(1, int(chromosome_index_size[chr_index_i]//25//batch_size - 1)):
			for batch_index_i in range(batch_size):
				batch_x[batch_index_i, :, :] = training_seq_input[:, (bp_index_i * 25 * batch_size - 500 + batch_index_i * 25):(bp_index_i * 25 * batch_size + 500 + batch_index_i * 25)]
			pred_feed_dict = {sequence: batch_x, cell: cell_id, assay: assay_id}
			prediction = sess.run([pred], feed_dict = pred_feed_dict)
			transformed = np.array(prediction, 'f')
			origin = (np.exp(transformed*2) - 1) / np.exp(transformed) / 2
			output[(bp_index_i*batch_size):((bp_index_i+1)*batch_size)] = np.reshape(origin, (1, batch_size))[0]
	print(time.time() - start)
	np.save(output_path + 'result' + '_' + args.chr_input + '_' + args.assay_cell, output)

if __name__=='__main__':
	main()


