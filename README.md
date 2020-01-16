# ENCODE Imputation Challenge
For a given cell type and assay, estimate the high throughput sequencing experiment results.
Our model: f(sequence, cell, assay)
convert_fa_onehotencode.py: input is the DNA sequence(hg19.fa, chromosome sepecific), output is the onehot encoded 4*n array.
train_cnn.py: tensorflow based CNN model.
predict_cnn.py: code for inference.
