# gigers_lstm
This project aims to apply unsupervised learning methods on text data and utilise the resulting model for some useful applications. The dataset is a list of music news titles. The applications are the following:

* unsupervised clustering (to be implemented)

* classification after an additional supervised training (to be implemented)

* robust searching in the text

* to tell if two article titles are about the same subject (to be implemented)

Two methods are tested:

1. The first one is an LSTM based language model. A General Recurrent Neural Networks is used, with one LSTM cell to predict the next word in a sentence, given the previous words. After training the model is capable of finishing the learned sentences, based on the first words fed into the model.

2. The second one is an LSTM based autoencoder. First the sentence is encoded with an LSTM encoding RNN, then the state of the RNN is fed through a deep autoencoder, then the resulting state is given to an LSTM decoding RNN which tries to reconstruct the input sentence. After training the model is capable of transforming a given sentence into a dense code, which can be used for the mentioned applications.

## Code
The code is separated to 4 parts:

1. word2vec_gigers.py: reads the raw text data and creates a dictionary and the embeddings lookup table from it, using skip-gram model.

2. gigers_dataset_to_embeddings.py: reads the raw text data and converts the words to embeddings, using the embeddings lookup table and the dictionary.

3. lstm_gigers.py: reads the embedded sentences, then trains and tests the LSTM language model.

4. lstm_autoencoder_gigers.py reads the embedded sentences, then trains and tests the LSTM autoencoder.

## Data
The dataset is a list of music news titles, collected from different blogs. The processed data files and a trained TensorFlow session is uploaded as well, completing the list of data files:

1. gigers_dataset.txt: raw text data of music news titles.

2. gigers_dataset_dictionary_embedding.pickle: pickle file containing the embedding lookup table, and the dictionary necessary to embed the text data. This file is the output of word2vec_gigers.py

3. gigers_dataset_embedded_100.pickle: 100 titles embedded to vectors. This file is the output of gigers_dataset_to_embeddings.py.

4. gigers_dataset_embedded_1000.zip: zipped pickle of 1000 titles embedded to vectors. This file is the output of gigers_dataset_to_embeddings.py.

5. 1000_news_learned_deep.ckpt: a trained session for the LSTM autoencoder model, using 1000 news as training data.

6. 1000_news_learned_deep.ckpt.meta meta graph for the trained session for the LSTM autoencoder model, using 1000 news as training data.

## Results

### LSTM Language Model
Results after 100 news titles are 'learned':

#### Generate sentence from input words
Original title: <br />
flashback foo fighters cover tom pettys breakdown UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK

Input words: <br />
flashback <br />
Predicted 6 further words: <br />
flashback watch randy travis duet with the

Input words: <br />
flashback foo fighters <br />
Predicted 4 further words: <br />
flashback watch fighters cover tom pettys breakdown

Input words: <br />
flashback foo fighters cover tom pettys <br />
Predicted 1 further word: <br />
flashback watch fighters cover tom pettys breakdown

### LSTM Autoencoder
These examples are the results of 1000 'learned' titles:

#### Generate sentence from input words

Input words: <br />
flashback foo fighters cover tom pettys <br />
Generated sentence: <br />
flashback foo fighters cover tom pettys breakdown burridge year usher arrests psa form watch listen berkeley ways us listen dacus listen stream watch to premiere rebels listen listen listen watch watch listen announcements co-owner anymore listen bittah 

Input words: <br />
flashback foo fighters tom pettys <br />
Generated sentence: <br />
flashback foo fighters cover tom pettys rejected connors hiring with stir concrete led endorsing concert-goers watch listen listen premiere manilow UNK the the spotifys listen listen premiere watch torches stream you ep to premiere listen listen watch

Input words: <br />
foo fighters tom pettys <br />
Generated sentence: <br />
exclusive grohl reveal underwood shoe crowd-funding eprom ernst with warned sumac watch falside stream 2014 premiere listen watch the involved stream premiere kygo watch you four-night iamnodobi harmonies listen listen bastille to sulaiman touche set listen listen 

Input words: <br />
foo fighters <br />
Generated sentence: <br />
guantanamera corporate lempel encased woodstock fighters with 11/20 of renaissance aarp rolled artwork listen ep hear scotty listen heart additions listen stream premiere centric 90 it to kidx stream on listen stream watch listen ep stream to

#### Search in the news

Search in the 1000 news titles and return the most relevant ones. The method is the following: <br />
All 1000 titles are encoded using the fully trained autoencoder. Then the search expression is encoded as well, the encoded search expression is compared to all the encoded titles, and the closest ones (cosine similarity) are selected as results.

Search expression: <br />
taylor swift shake it off <br />
Search results (top 7): <br />
'kendrick lamar sings freestyles over taylor swifts shake it off' <br />
'kendrick lamar sings taylor swift s shake it off   listen' <br />
'taylor swifts shake it off returns to no 1 on hot 100' <br />
'taylor swift shuns grand experiment of streaming music' <br />
'maison cartel   obsidian original mix free download' <br />
'kendrick lamar & taylor swift shout out each other s songs audio/video' <br />
'taylor swift collects fourth no 1 album now 52 debuts at no 2'

Search expression: <br />
ac/dc drummer murder <br />
Search results (top 7): <br />
'ac/dc drummer charged with hiring hitman to murder two people' <br />
'ac/dc drummer charged in murder plot' <br />
'hip-hop rumors did tiny join two black sororities pics' <br />
'hip-hop rumors can biggie s son spit or nah' <br />
'bono defends spotify lets experiment lets see what works' <br />
'jay z acquires armand de brignac' <br />
'metal injection livecast #280   metallicas some kind of monster dvd running commentary' <br />

Search expression: <br />
run the jewels 2 <br />
Search results (top 7): <br />
'run the jewels 2' <br />
'run the jewels run the jewels 2' <br />
'run the jewels run the jewels 2' <br />
'run the jewels deliver earth-shattering performance of early on letterman   watch' <br />
'run the jewels close your eyes and count to fuck ft zack de la rocha' <br />
'run the jewels how 2014s brashest rap duo came back from oblivion' <br />
'ought more than any other day' <br />

Search expression: <br />
jay z buys champagne brand <br />
Search results (top 7): <br />
'jay z buys ace of spades champagne brand' <br />
'jay z had a favorite champagne brand so he bought it' <br />
'jay z acquires armand de brignac' <br />
'jay z s favorite artist has a son that raps' <br />
'a hooligans az elso idei comet gyoztes' <br />
'zac brown on new album our boundaries have dissolved' <br />
'hip-hop rumors did tiny join two black sororities pics'
