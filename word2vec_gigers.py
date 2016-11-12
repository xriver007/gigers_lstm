
# coding: utf-8

# In[13]:

from __future__ import absolute_import
from __future__ import print_function

import collections
import math
import os
import random
import zipfile

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import pickle

#based on: https://github.com/tensorflow/tensorflow/blob/r0.11/tensorflow/examples/tutorials/word2vec/word2vec_basic.py

############################
#load and process text data#
############################

#load text as data
with open ("gigers_dataset.txt", "r") as datat_text_file:
    data_text = datat_text_file.readlines()
    datat_text_file.close()
data_text = data_text[0]

#replace line separators markers with spaces
data_text = data_text.replace('<titleSeparator>', ' ')

#remove special characters
def remove_non_ascii_128(text):
    return ''.join([i if ord(i) < 128 else ' ' for i in text])
data_text = remove_non_ascii_128(data_text)

#remove some further undesirable characters
def remove_some_chars(text):
    chars_to_remove = set(['[', ']', '(', ')', "'", '"', '!', '?', '.', ',', '|', '@'])
    return ''.join([i for i in text if i not in chars_to_remove])
data_text = remove_some_chars(data_text)

#to lower case
data_text = data_text.lower()

words = data_text.split()

print('number of words: ', len(words))


#build the dictionary and replace rare words with UNK token
vocabulary_size = 50000 #number of words we want to include to dictionary (most frequent words)

#selects the vocabulary_size most common words in the dataset (words)
#creates dictionary to map words to ids (UNK has id of 0, most frequent word has id of 1 and so on)
#creates reverse_dictionary to map ids to words
def build_dataset(words):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(words)
del words  #to reduce memory.
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10])


################
#generate batch#
################

data_index = 0

#generates the (target -> context) pairs
#target words: batces (shape = (batch_size,))
#context words: labels (shape = (batch_size,1))
#as labels, generate_batch selects one word from the skip_window sized environment of the target word
#but for one target word, this happens num_skips times, so evantually for one target word num_skips number of context word is selected
def generate_batch(batch_size, num_skips, skip_window):
  global data_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * skip_window + 1 # [ skip_window target skip_window ]
  buffer = collections.deque(maxlen=span)
  for _ in range(span):
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  for i in range(batch_size // num_skips):
    target = skip_window  # target label at the center of the buffer
    targets_to_avoid = [ skip_window ]
    for j in range(num_skips):
      while target in targets_to_avoid:
        target = random.randint(0, span - 1)
      targets_to_avoid.append(target)
      batch[i * num_skips + j] = buffer[skip_window]
      labels[i * num_skips + j, 0] = buffer[target]
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  return batch, labels

batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
  print(batch[i], '->', labels[i, 0])
  print(reverse_dictionary[batch[i]], '->', reverse_dictionary[labels[i, 0]])

    
#################
#hyperparameters#
#################

batch_size = 128
embedding_size = 128  #dimension of the embedding vector
skip_window = 1       #how many words to consider left and right
num_skips = 2         #how many times to reuse an input to generate a label
num_sampled = 64      #number of negative examples to sample

#pick a random validation set to sample nearest neighbors
#select from the most frequent words
valid_size = 16     #wandom set of words
valid_window = 100  #from the 100 most frequent words
valid_examples = np.array(random.sample(range(valid_window), valid_size)) #ids of the valid example words


num_steps = 100001


#################
#skip-gram model#
#################

graph = tf.Graph()

with graph.as_default():

  #input data placeholders and variables
  train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
  train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
  valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

  #ops and variables pinned to the CPU because of missing GPU implementation
  with tf.device('/cpu:0'):
    #variable for embeddings
    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    
    #look up embeddings for train inputs
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    #construct the variables for the noise-contrastive estimation (NCE) loss
    nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

  #compute the average NCE loss for the batch
  #tf.nn.nce_loss automatically draws a new sample of the negative labels each
  loss = tf.reduce_mean(
      tf.nn.nce_loss(nce_weights, nce_biases, embed, train_labels,
                     num_sampled, vocabulary_size))

  #construct the SGD optimizer using a learning rate of 1.0.
  optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

  
  norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
  normalized_embeddings = embeddings / norm

  #get validation embeddings from embedding example ids
  valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
  #compute the cosine similarity between validation embedings and all embeddings.
  similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)


##########
#training#
##########

with tf.Session(graph=graph) as session:
  #initialize variables
  tf.initialize_all_variables().run()
  print("Initialized")

  average_loss = 0
  for step in xrange(num_steps):
    
    #get training batch
    batch_inputs, batch_labels = generate_batch(
        batch_size, num_skips, skip_window)
    feed_dict = {train_inputs : batch_inputs, train_labels : batch_labels}

    #training step
    _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
    average_loss += loss_val

    #print average words of the
    if step % 2000 == 0:
      if step > 0:
        average_loss /= 2000
      # The average loss is an estimate of the loss over the last 2000 batches.
      print("Average loss at step ", step, ": ", average_loss)
      average_loss = 0

    #print validation words and similar words to it
    if step % 10000 == 0:
      sim = similarity.eval()
      for i in xrange(valid_size):
        valid_word = reverse_dictionary[valid_examples[i]]
        top_k = 8 # number of nearest neighbors
        nearest = (-sim[i, :]).argsort()[1:top_k+1]
        log_str = "Nearest to %s:" % valid_word
        for k in xrange(top_k):
          close_word = reverse_dictionary[nearest[k]]
          log_str = "%s %s," % (log_str, close_word)
        print(log_str)
  

  #get final embedding
  final_embeddings = normalized_embeddings.eval()





# In[14]:



save_pickle = True

if save_pickle:
    # Saving the objects:
    with open('gigers_dataset_dictionary_embedding.pickle', 'wb') as f:  # Python 3: open(..., 'wb'), Python 3: open(..., 'w')
        pickle.dump([final_embeddings, dictionary, reverse_dictionary], f)


# In[ ]:



