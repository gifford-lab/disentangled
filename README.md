# disentangled

To run the experiment, use python3 SeqGen_v4.py $HIDDEN $LEN $UNITS $EPOCHS $SAVE

Where:

HIDDEN= number of neurons to use in the hidden layers of the autoencoder (the encoder and decoder have the same intermediate size)

LEN = PCA dimension

UNITS = number of latent neurons assigned to each disentangled dimension

EPOCHS = how many epochs to train

SAVE = file to save the results to.

There are some settings in SeqGen_v4.py you can play around with at the top. We include a sample.csv file to indicate the format of the input data.
