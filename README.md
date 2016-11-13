# gigers_lstm
This is an LSTM based language model. After training the model is capable of finishing the learned sentences, based on the first words fed in to the model.

The code is separeted to 3 parts:
1. word2vec_gigers.py: reads the raw text data and creates a dictionary and the embeddings lookup table from it using, using skip-gram model.
2. gigers_dataset_to_embeddings.py: reads the raw text data and converts the words to embeddings, using the embeddings lookup table and the dictionary.
3. lstm_gigers.py: reads the embedded sentences and trains an LSTM language model.

The data is a list of music news titles, collected from different blogs. The processd data files are uploaded as well, completing the list of data files:
1. gigers_dataset.txt: raw text data of music news titles.
2. gigers_dataset_dictionary_embedding.pickle: pickle file containing the embedding lookup table, and the dictionary necessary to embed the text data. This file is the output of word2vec_gigers.py
3. gigers_dataset_embedded.pickle: the titles embedded to vectors. This file is the output of gigers_dataset_to_embeddings.py.
