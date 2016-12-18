# gigers_lstm
This project aims to apply unsupervised learning methods on text data and utilise the pretrained model for some useful applications. The dataset is a list of music news headlines. The applications re the following:

* unsupervised clustering (to be implemented)

* classification after an additional supervised training (to be implemented)

* robust searching in the text

* to tell if two article titles are about the same subject (to be implemented)

Two methods are tested:

1. The first one is an LSTM based language model. A General Recurrent Neural Networks is used, with one LSTM cell to predict the next word in a sentence, given the previous words. After training the model is capable of finishing the learned sentences, based on the first words fed into the model.

2. The second one is an LSTM based autoencoder. First the sentence is encoded with an LSTM encoding RNN, then the state of the RNN is fed through an autoencoder, then the resulting state is given to an LSTM decoding RNN which tries to reconstruct the input sentence. After training the model is capable of representing the sentence to a dense code, which can be used for the mentioned applications.

## Code
The code is separated to 3 parts:

1. word2vec_gigers.py: reads the raw text data and creates a dictionary and the embeddings lookup table from it, using skip-gram model.

2. gigers_dataset_to_embeddings.py: reads the raw text data and converts the words to embeddings, using the embeddings lookup table and the dictionary.

3. lstm_gigers.py: reads the embedded sentences and trains an LSTM language model.

## Data
The data is a list of music news titles, collected from different blogs. The processed data files are uploaded as well, completing the list of data files:

1. gigers_dataset.txt: raw text data of music news titles.

2. gigers_dataset_dictionary_embedding.pickle: pickle file containing the embedding lookup table, and the dictionary necessary to embed the text data. This file is the output of word2vec_gigers.py

3. gigers_dataset_embedded.pickle: the titles embedded to vectors. This file is the output of gigers_dataset_to_embeddings.py.

## Results
Here are some examples after 36000 training steps on 100 different news titles.

#### Predictions at the end of the training
Original sentence: <br />
flashback foo fighters cover tom pettys breakdown UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK <br />
Supported prediction: <br />
flashback watch fighters cover tom pettys breakdown gets the diana jame$ the brent the watch the watch <br />
Standalone prediction: <br />
flashback watch randy travis duet with the UNK the UNK the UNK UNK UNK UNK UNK UNK 

Original sentence: <br />
kanye west miguel to appear on lordes hunger games soundtrack UNK UNK UNK UNK UNK UNK UNK <br />
Supported prediction: <br />
kanye west surprisingly to appear on lordes hunger games soundtrack stream his the auditioned the the kids <br />
Standalone prediction: <br />
kanye west surprisingly tones down onstage rant UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK

#### Using the model
Original sentence: <br />
flashback foo fighters cover tom pettys breakdown UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK <br />
Fed in to the model: <br />
flashback <br />
Predicted 6 further words: <br />
flashback watch randy travis duet with the <br />
Fed in to the model: <br />
flashback foo fighters <br />
Predicted 4 further words: <br />
flashback watch fighters cover tom pettys breakdown <br />
Fed in to the model: <br />
flashback foo fighters cover tom pettys <br />
Predicted 1 further word: <br />
flashback watch fighters cover tom pettys breakdown
