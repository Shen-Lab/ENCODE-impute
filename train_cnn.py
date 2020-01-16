import numpy as np
import tensorflow as tf
from os import listdir
from os.path import isfile, join

path = '/scratch/user/wuwei_tan/imputation_challenge/'

seq_path = path + 'data/chr_npy/'
model_path = path + 'model/cnn_model/'
training_label_path = path + 'pilot_region_label/training_aver_25_label_transformed/'
validation_label_path = path + 'pilot_region_label/training_aver_25_label_transformed/'

chromosome_index = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 'chr20', 'chr21', 'chr22', 'chrX']
chromosome_index_size = np.array((248956422, 242193529, 198295559, 190214555, 181538259, 170805979, 159345973, 145138636, 138394717, 133797422, 135086622, 133275309, 114364328, 107043718, 101991189, 90338345, 83257441, 80373285, 58617616, 64444167, 46709983, 50818468, 156040895))


cell_embedding_size = 20
assay_embedding_size = 12

batch_size = 256



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

chr_index_range = np.arange(len(chromosome_index)
with tf.Session() as sess:
	for training_epoch in range(100):
		random.shuffle(chr_index_range)	
		for chr_index_i in range(len(chromosome_index)):
			training_label_path_temp = training_label_path + chromosome_size[chr_index_range[chr_index_i]] + '/'
			training_files = [f for f in listdir(training_label_path_temp) if isfile(join(training_label_path_temp, f))]
			for training_files_i in range(len(training_files)):
				training_seq_input = np.load(seq_path + chromosome_index[chr_index_range[chr_index_i]] + '.npy')
				cell_id = np.repeat(int(training_files[training_files_i][1:3]) - 1, len(training_seq_input))
				assay_id = np.repeat(int(training_files[training_files_i][4:6]) - 1, len(training_seq_input))
				train_feed_dict = {sequence: , cell: cell_id, assay: assay_id}
				training_loss = sess.run([loss_op], feed_dict = train_feed_dict)
			validation_label_path_temp = validation_label_path + chromosome_size[chr_index_range[chr_index_i]] + '/'
			validation_files = [f for f in listdir(validation_label_path_temp) if isfile(join(validation_label_path_temp, f))]
			for validation_files_i in range(len(validation_files)):
				validation_seq_input = np.load(seq_path + chromosome_index[chr_index_range[chr_index_i]] + '.npy')
				cell_id = np.repeat(int(training_files[validation_files_i][1:3]) - 1, len(validation_seq_input))
				assay_id = np.repeat(int(training_files[validation_files_i][4:6]) - 1, len(validation_seq_input))
				valid_feed_dict = {sequence: , cell: cell_id, assay: assay_id}
				validation_loss = sess.run([loss_mse], feed_dict = valid_feed_dict)
				validation_loss = np.mean(np.asarray(validation_loss))
			with open(path + 'log.txt', 'a+') as log_file:
				log_file.writelines('Epoch ' + str(training_epoch) + ' ' + chromosome_index[chr_index_range[chr_index_i]] + ' , training_loss = ' + {:.4f}.format(training_loss))
				log_file.writelines('Epoch ' + str(training_epoch) + ' ' + chromosome_index[chr_index_range[chr_index_i]] + ' , training_loss = ' + {:.4f}.format(validation_loss))
		saver.save(sess, model_path + 'model'+'_'+str(training_epoch))

