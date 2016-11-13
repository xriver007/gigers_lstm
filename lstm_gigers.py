#based on https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/udacity/6_lstm.ipynb

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import random
import pickle
import time


#####################
#load data and batch#
#####################

#Load data (embeddings, dictionary and reverse dictionary)
with open('gigers_dataset_embedded.pickle') as f:  # Python 3: open(..., 'rb')
    sentence_list_embedded = pickle.load(f)

#load embeddings
with open('gigers_dataset_dictionary_embedding.pickle') as f:  # Python 3: open(..., 'rb')
    final_embeddings, dictionary, reverse_dictionary = pickle.load(f)

vocabulary_size, embedding_size = final_embeddings.shape
    
#port data to name used later
sentence_list = sentence_list_embedded

print('Number of sentences: ' + str(len(sentence_list)))
    
def next_batch(sentence_list, num_unrollings, index = None):
    #batch size is 1 for now
    batch_size = 1
    sentence_list_length = len(sentence_list)
    if index == None:
        index = random.randint(0,sentence_list_length-1)
    selected_sentence = sentence_list[index]
    batches = []
    for i in range(num_unrollings + 1):
        reshape_dim = len(selected_sentence[i])
        batches.append(np.reshape(selected_sentence[i], (1,reshape_dim)))
    return batches


###################
#tuning parameters#
###################

learning_rate = 0.004
input_dimension = len(sentence_list[0][0]) #128
cell_size = 1024
num_unrollings = len(sentence_list[0])-1
batch_size = 1
is_standalone_loss = False
is_softmax_loss = False
is_adam_optimizer = True

step_number = 36000
log_step_number = 500


#######
#model#
#######

#input gate: input, previous output, and bias
ix = tf.Variable(tf.truncated_normal([input_dimension, cell_size], -0.1, 0.1))
im = tf.Variable(tf.truncated_normal([input_dimension, cell_size], -0.1, 0.1))
ib = tf.Variable(tf.zeros([1, cell_size]))
#forget gate: input, previous output, and bias
fx = tf.Variable(tf.truncated_normal([input_dimension, cell_size], -0.1, 0.1))
fm = tf.Variable(tf.truncated_normal([input_dimension, cell_size], -0.1, 0.1))
fb = tf.Variable(tf.zeros([1, cell_size]))
#memory cell: input, state and bias
cx = tf.Variable(tf.truncated_normal([input_dimension, cell_size], -0.1, 0.1))
cm = tf.Variable(tf.truncated_normal([input_dimension, cell_size], -0.1, 0.1))
cb = tf.Variable(tf.zeros([1, cell_size]))
#output gate: input, previous output, and bias
ox = tf.Variable(tf.truncated_normal([input_dimension, input_dimension], -0.1, 0.1))
om = tf.Variable(tf.truncated_normal([input_dimension, input_dimension], -0.1, 0.1))
ob = tf.Variable(tf.zeros([1, input_dimension]))
#variables to map states to output
w = tf.Variable(tf.truncated_normal([cell_size, input_dimension], -0.1, 0.1))
b = tf.Variable(tf.zeros([input_dimension]))
#variables saving state across unrollings
saved_output_supported = tf.Variable(tf.zeros([batch_size, input_dimension]), trainable=False)
saved_state_supported = tf.Variable(tf.zeros([batch_size, cell_size]), trainable=False)
saved_output_standalone = tf.Variable(tf.zeros([batch_size, input_dimension]), trainable=False)
state_predict_saved = tf.Variable(tf.zeros([batch_size, cell_size]), trainable=False)


#LSTM cell
def lstm_cell(i, o, state):
    input_gate = tf.sigmoid(tf.matmul(i, ix) + tf.matmul(o, im) + ib)
    forget_gate = tf.sigmoid(tf.matmul(i, fx) + tf.matmul(o, fm) + fb)
    update = tf.matmul(i, cx) + tf.matmul(o, cm) + cb
    state = forget_gate * state + input_gate * tf.tanh(update)
    output_gate = tf.sigmoid(tf.matmul(i, ox) + tf.matmul(o, om) + ob)
    output = tf.nn.l2_normalize(output_gate * tf.nn.xw_plus_b(state, w, b), dim=1)
    return output, state

#variable for training data
train_data = list()
for _ in range(num_unrollings + 1):
    train_data.append(tf.placeholder(tf.float32, shape=[batch_size,input_dimension]))
train_inputs = train_data[:num_unrollings]
train_labels = train_data[1:]  # labels are inputs shifted by one time step.

#for softmax loss
softmax_train_labels = tf.nn.softmax(tf.concat(0, train_labels))


output_supported = saved_output_supported
state_supported = saved_state_supported

#enroll supported
outputs_supported = list()
for i in train_inputs:
    output_supported, state_supported = lstm_cell(i, output_supported, state_supported)
    outputs_supported.append(output_supported)

#loss supported
with tf.control_dependencies([saved_output_supported.assign(output_supported), saved_state_supported.assign(state_supported)]):
    outputs_supported = tf.concat(0, outputs_supported)

    if is_softmax_loss:
        #calculate loss as the cross entropy of the softmax of the predicted and training words
        softmax_supported = tf.nn.softmax(outputs_supported)
        loss_supported = -tf.reduce_sum(softmax_train_labels*tf.log(softmax_supported))
    else:
        #calculate loss as the negative average cosine distance of the predicted and training words
        loss_supported = -tf.reduce_mean(tf.mul(tf.reshape(outputs_supported, [-1]), tf.reshape(tf.concat(0, train_labels), [-1])))
    
output_standalone = saved_output_standalone
state_standalone = state_predict_saved

#enroll standalone
outputs_standalone = list()
for index, i in enumerate(train_inputs):
    if index == 0:
        output_standalone, state_standalone = lstm_cell(i, output_standalone, state_standalone)
    else:
        output_standalone, state_standalone = lstm_cell(output_standalone, output_standalone, state_standalone)
    outputs_standalone.append(output_standalone)

#loss standalone
with tf.control_dependencies([saved_output_standalone.assign(output_standalone), state_predict_saved.assign(state_standalone)]):
    outputs_standalone = tf.concat(0, outputs_standalone)

    if is_softmax_loss:
        #calculate loss as the cross entropy of the softmax of the predicted and training words
        softmax_standalone = tf.nn.softmax(outputs_standalone)
        loss_standalone = -tf.reduce_sum(softmax_train_labels*tf.log(softmax_standalone))
    else:
        #calculate loss as the negative average cosine distance of the predicted and training words
        loss_standalone = -tf.reduce_mean(tf.mul(tf.reshape(outputs_standalone, [-1]), tf.reshape(tf.concat(0, train_labels), [-1])))


if is_standalone_loss:
    loss = loss_standalone
else:
    loss = loss_supported

#optimizer
if is_adam_optimizer:
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
else:
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)


##########
#training#
##########

start_time = time.time()

#create session
session = tf.Session()

#initialize session
init = tf.initialize_all_variables()
session.run(init)

#for loss logging
losses = [] #the loss with the actual training sample
losses_sample = [] #the average of the loss of two sample sentence
losses_0 = []
losses_1 = []

for i in range(step_number):

    #create feed_dict for this step
    batches = next_batch(sentence_list, num_unrollings)
    feed_dict = dict()
    for j in range(num_unrollings + 1):
        feed_dict[train_data[j]] = batches[j]

    #train step
    session.run(train_step, feed_dict=feed_dict)

    #log loss and prediction
    if (i % log_step_number) == 0:
        #loss of the actual training sample
        loss_out = session.run(loss, feed_dict=feed_dict)
        losses.append(loss_out)
        
        #sample 0
        batches = next_batch(sentence_list, num_unrollings, 0)
        feed_dict = dict()
        for j in range(num_unrollings + 1):
            feed_dict[train_data[j]] = batches[j]
        loss_out_0 = session.run(loss, feed_dict=feed_dict)
        losses_0.append(loss_out_0)

        #sample 1
        batches = next_batch(sentence_list, num_unrollings, 1)
        feed_dict = dict()
        for j in range(num_unrollings + 1):
            feed_dict[train_data[j]] = batches[j]
        loss_out_1 = session.run(loss, feed_dict=feed_dict)
        losses_1.append(loss_out_1)
        
        #average loss of the two sample
        losses_sample.append((loss_out_0 + loss_out_1) / 2)

        #print progress
        print(str((float(i) / step_number)*100) + '%')

 
      
training_time = time.time() - start_time
print('Training time: ' + str(training_time))


#######
#plots#
#######

print('Trained loss: ' + str(losses_sample[-1])) 
plt.plot(losses_sample) #blue
plt.plot(losses) #green


#################
#save or restore#
#################

#saver = tf.train.Saver()
#save_path = saver.save(session, "100_news_learned.ckpt")
#print("Model saved in file: %s" % save_path)
#saver.restore(session, "100_news_learned.ckpt")
#print("Model restored from file: %s" % save_path)


#########################
#print training examples#
#########################

#predictions for the two samples
#sample 0
batches = next_batch(sentence_list, num_unrollings, 0)
feed_dict = dict()
for j in range(num_unrollings + 1):
    feed_dict[train_data[j]] = batches[j]
outputs_standalone_out_0, outputs_supported_out_0 = session.run([outputs_standalone, outputs_supported], feed_dict=feed_dict)
losses_0.append(loss_out_0)

#sample 1
batches = next_batch(sentence_list, num_unrollings, 1)
feed_dict = dict()
for j in range(num_unrollings + 1):
    feed_dict[train_data[j]] = batches[j]
outputs_standalone_out_1, outputs_supported_out_1 = session.run([outputs_standalone, outputs_supported], feed_dict=feed_dict)
losses_1.append(loss_out_1)

#add first word to the predicted sentence
predicted_sentence_supported_0 = np.vstack([sentence_list[0][0], outputs_supported_out_0])
predicted_sentence_standalone_0 = np.vstack([sentence_list[0][0], outputs_standalone_out_0])

predicted_sentence_supported_1 = np.vstack([sentence_list[1][0], outputs_supported_out_1])
predicted_sentence_standalone_1 = np.vstack([sentence_list[1][0], outputs_standalone_out_1])


def sentence_embedded_to_string(embeddings, sentence_embedded):
    dist = np.dot(embeddings, np.transpose(sentence_embedded))
    #dist_max = np.amax(dist, axis=0)
    word_indeces = np.argmax(dist, axis=0)
    sentence_string = ''
    for word_index in word_indeces:
        sentence_string += reverse_dictionary[word_index] + ' '
    return sentence_string
    
print('Original sentence: ')
print(sentence_embedded_to_string(final_embeddings, sentence_list[0]))
print('Supported prediction: ')
print(sentence_embedded_to_string(final_embeddings, predicted_sentence_supported_0))
print('Standalone prediction: ')
print(sentence_embedded_to_string(final_embeddings, predicted_sentence_standalone_0))

print('Original sentence: ')
print(sentence_embedded_to_string(final_embeddings, sentence_list[1]))
print('Supported prediction: ')
print(sentence_embedded_to_string(final_embeddings, predicted_sentence_supported_1))
print('Standalone prediction: ')
print(sentence_embedded_to_string(final_embeddings, predicted_sentence_standalone_1))


###############################
#use model to finish sentences#
###############################

#generate input data from string
prediction_length = 4

sentence_beggining = "flashback foo fighters"
sentence_beggining_words = sentence_beggining.split()
prediction_input_length = len(sentence_beggining_words)

sentence_embedded = np.zeros((prediction_input_length, embedding_size))
for index_in_sentence, word in enumerate(sentence_beggining_words):
    try:
        word_index = dictionary[word]
    except:
        word_index = 0
    sentence_embedded[index_in_sentence] = final_embeddings[word_index]

#get it to the right format
sentence_embedded_list = list()
sentence_embedded_list.append(sentence_embedded)
sentence_embedded_batch = next_batch(sentence_embedded_list, prediction_input_length-1)
  
    
#update graph

embeddings = tf.constant(final_embeddings)

output_predict = tf.Variable(tf.zeros([1, input_dimension]), trainable=False)
state_predict = tf.Variable(tf.zeros([1, cell_size]), trainable=False)

output_predict = tf.zeros([1, input_dimension])
state_predict = tf.zeros([1, cell_size])

prediction_inputs = list()
for _ in range(prediction_input_length):
    prediction_inputs.append(tf.placeholder(tf.float32, shape=[1,input_dimension]))
    
outputs_predict = list()
states_predict = list()

#enroll with the sentence beggining and after that enroll prediction_length times
for i in range(prediction_input_length + prediction_length - 1):
    if i<prediction_input_length:
        output_predict, state_predict = lstm_cell(prediction_inputs[i], output_predict, state_predict)
        outputs_predict.append(output_predict)
    else:
        output_predict, state_predict = lstm_cell(output_predict, output_predict, state_predict)

        outputs_predict.append(output_predict)
        states_predict.append(state_predict)

outputs_predict = tf.concat(0, outputs_predict)  
states_predict = tf.concat(0, states_predict)
   
    
#run session   
for j in range(prediction_input_length):
    feed_dict[prediction_inputs[j]] = sentence_embedded_batch[j]
outputs_predict_out = session.run(outputs_predict, feed_dict=feed_dict)

print('Feed in:')
print(sentence_embedded_to_string(final_embeddings, sentence_embedded))
print('Predicted:')
print(sentence_embedded_to_string(final_embeddings, outputs_predict_out))

