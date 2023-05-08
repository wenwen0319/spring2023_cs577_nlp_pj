The dependency_graph_DE.py will read in the data, sample the random walks and  do the distance encoding part. For each aspect and it's context, the file will change it into a number(128) of sequences.

It requires Spacy to run.

The train_DE.py is the main function. Just run it, use --dataset to indicate the dataset(rest14, rest15, rest16, twitter, lap14), --model_name to indicate different models(DE, DE_LSTM, DE_LSTM_exp).

It requires pytorch, sklearn, numpy to run.

To run the code, you also needs to download the glove embedding, and change the path in the data_utils_DE.py line 33 fname = './{YOUR_FILE_PATH}'.

More detailed dataset information can be found here https://github.com/GeneZC/ASGCN . We mainly follow the setting in this link.
