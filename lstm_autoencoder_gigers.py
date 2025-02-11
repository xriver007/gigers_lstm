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


#embeddings#
############

#load embeddings dictionary and reverse dictionary
with open('gigers_dataset_dictionary_embedding.pickle') as f:  # Python 3: open(..., 'rb')
    final_embeddings, dictionary, reverse_dictionary = pickle.load(f)

vocabulary_size, embedding_size = final_embeddings.shape


#unlabelled data for autoencoder#
#################################

#load embedded text data
with open('gigers_dataset_embedded_1000.pickle') as f:  # Python 3: open(..., 'rb')
    sentence_list_embedded, sentence_list_string = pickle.load(f)
    #sentence_list_embedded, sentence_list_string are the embedded and string versions of the news list
    
#port data to name used later
sentence_list = sentence_list_embedded

print('Number of sentences: ' + str(len(sentence_list)))


#labelled data for classification#
##################################

#load labelled data for classification
with open('music_news_dataset_labelled_embedded.pickle') as f:  # Python 3: open(..., 'rb')
    sentence_list_embedded_labelled, sentence_list_string_labelled, labels = pickle.load(f)
    #labels: [1 0] -> not new muic, [0 1] -> new music
    
data_number = len(sentence_list_embedded_labelled)
training_number = int(data_number*0.7)

#generate training indeces
random.seed(10)
training_indeces = random.sample(range(len(sentence_list_embedded_labelled)), training_number)
random.seed()


#separate training and test subsets (this takes long time)
training_embedded = [sentence_list_embedded_labelled[index] for index in training_indeces]
test_embedded = [sentence_list_embedded_labelled[index] for index, _ in enumerate(sentence_list_embedded_labelled) if index not in training_indeces]

training_string = [sentence_list_string_labelled[index] for index in training_indeces]
test_string = [sentence_list_string_labelled[index] for index, _ in enumerate(sentence_list_embedded_labelled) if index not in training_indeces]

training_labes = [labels[index] for index in training_indeces]
test_labes = [labels[index] for index, _ in enumerate(sentence_list_embedded_labelled) if index not in training_indeces]


#batch generator#
#################

#batch generator function
def next_batch(sentence_list, num_unrollings, labels = None, batch_size = 1, index = None):
    
    sentence_list_length = len(sentence_list)
    
    #select sntences
    selected_sentences = []
    data_indeces = []
    for i in range(batch_size):
        if index == None:
            index_selected = random.randint(0,sentence_list_length-1)
        else:
            index_selected = index
        
        selected_sentences.append(sentence_list[index_selected])
        data_indeces.append(index_selected)
    
    #reshape data and create batches
    data_batch = []
    embedding_size = len(selected_sentences[0][0])
    for i in range(num_unrollings + 1):
        reshaped_word_bach = np.zeros((batch_size,embedding_size))
        for j in range(batch_size):
            text_length = len(selected_sentences[j])
            reshaped_word = np.reshape(selected_sentences[j][i], (1,embedding_size))
            reshaped_word_bach[j,:] = reshaped_word
        data_batch.append(reshaped_word_bach)
        
    #labels
    if labels != None:
        label_number = len(labels[0])
        labels_batch = np.zeros((batch_size,label_number))
        for j in range(batch_size):
            label_reshaped = np.reshape(labels[data_indeces[j]], (1,label_number))
            labels_batch[j,:] = label_reshaped
    else:
        labels_batch = None
    
    
    return data_batch, labels_batch


###################
#tuning parameters#
###################

learning_rate = 0.004
input_dimension = len(sentence_list[0][0]) #128
cell_size = 1024
hidden_dim_1 = 512
hidden_dim_2 = 256
code_size = hidden_dim_2
num_unrollings = len(sentence_list[0])-1
batch_size = 64
is_softmax_loss = False
is_adam_optimizer = True
is_load_trained_autoencoder = True
is_save_session = False
step_number = 14000
log_step_number = 100

learning_rate_class = 0.004
step_number_class = 10000
log_step_number_class = 10
log_step_number_accuracy = 400
labels_number = len(labels[0])
keep_prob_class = 0.5
is_load_trained_classifier = True


#######
#model#
#######


#LSTM and variables#
####################

#LSTM class (variables and computation)
class LstmCell:
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
    #output and state variables
    output = tf.Variable(tf.zeros([batch_size, input_dimension]), trainable=False)
    state = tf.Variable(tf.zeros([batch_size, cell_size]), trainable=False)


    #make one step
    def run(self, input_value):
        input_gate = tf.sigmoid(tf.matmul(input_value, self.ix) + tf.matmul(self.output, self.im) + self.ib)
        forget_gate = tf.sigmoid(tf.matmul(input_value, self.fx) + tf.matmul(self.output, self.fm) + self.fb)
        update = tf.matmul(input_value, self.cx) + tf.matmul(self.output, self.cm) + self.cb
        self.state = forget_gate * self.state + input_gate * tf.tanh(update)
        output_gate = tf.sigmoid(tf.matmul(input_value, self.ox) + tf.matmul(self.output, self.om) + self.ob)
        self.output = tf.nn.l2_normalize(output_gate * tf.nn.xw_plus_b(self.state, self.w, self.b), dim=1)
    
    #for the decoder to get the state from the encoder
    def set_state(self, input_state):
        self.state.assign(input_state)
        
#variable for hidden layer for deep encoder
w_hidden_1 = tf.Variable(tf.truncated_normal([cell_size, hidden_dim_1], -0.1, 0.1))
b_hidden_1 = tf.Variable(tf.zeros([hidden_dim_1]))
w_hidden_2 = tf.Variable(tf.truncated_normal([hidden_dim_1, hidden_dim_2], -0.1, 0.1))
b_hidden_2 = tf.Variable(tf.zeros([hidden_dim_2]))

w_hidden_1_prime = tf.transpose(w_hidden_1)  # tied weights between encoder and decoder
b_hidden_1_prime = tf.Variable(tf.zeros([cell_size]))
w_hidden_2_prime = tf.transpose(w_hidden_2)  # tied weights between encoder and decoder
b_hidden_2_prime = tf.Variable(tf.zeros([hidden_dim_1]))

#variables for classification
w_class = tf.Variable(tf.random_normal([code_size, labels_number]), name = 'w_class')
b_class = tf.Variable(tf.random_normal([labels_number]), name = 'b_class')


#placeholders#
##############

#placeholders for input sentence
x = list()
for _ in range(num_unrollings + 1):
    x.append(tf.placeholder(tf.float32, shape=[batch_size,input_dimension]))
    
#placeholder for classification training labels
y_class_ = tf.placeholder(tf.float32, [None, labels_number])


#build up model#
################

#encoder cell
encoder_cell = LstmCell()

#enroll encoder (feed every input through the encoder)
for i in x:
    encoder_cell.run(i)

#hidden layers, compressing the data
h1 = tf.nn.xw_plus_b(encoder_cell.state, w_hidden_1, b_hidden_1)
h2 = tf.nn.xw_plus_b(h1, w_hidden_2, b_hidden_2)

#the normalized encoded vector of the most dense data
code = tf.nn.l2_normalize(h2, dim=1)

#classifier layer
y_class = tf.nn.softmax(tf.nn.xw_plus_b(tf.nn.dropout(code, keep_prob = keep_prob_class), w_class, b_class))

#hidden layers, decompressing the data
h2_prime = tf.nn.xw_plus_b(h2, w_hidden_2_prime, b_hidden_2_prime)
h1_prime = tf.nn.xw_plus_b(h2_prime, w_hidden_1_prime, b_hidden_1_prime)
    
#decoder cell
decoder_cell = LstmCell()

#enroll decoder
y = list()
for index in range(num_unrollings + 1):
    if index == 0:
        #for the first enrolling, get the initial state from the decompressed data
        decoder_cell.set_state(h1_prime)
    encoder_cell.run(encoder_cell.output)
    #store the outputs to a list
    y.append(encoder_cell.output)
    
#list of tensors -> one tensor, shape = (batch_size * (num_unrollings + 1), embedding_size)
y_concat = tf.concat(0, y)
x_concat = tf.concat(0, x)


#losses#
########

#loss calculations
if is_softmax_loss:
    #calculate loss as the cross entropy of the softmax of the predicted and training words
    x_softmax = tf.nn.softmax(x_concat)
    y_softmax = tf.nn.softmax(y_concat)
    loss = -tf.reduce_sum(x_softmax*tf.log(y_softmax))
else:
    #calculate loss as the negative average cosine distance of the predicted and training words
    loss = -tf.reduce_mean(tf.mul(tf.reshape(y_concat, [-1]), tf.reshape(x_concat, [-1])))


#loss function for classifier (cross entropy)
loss_class = -tf.reduce_sum(y_class_*tf.log(y_class))


#optimizers#
############

#optimizer for autoencoder
if is_adam_optimizer:
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
else:
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
        
#optimizer for classifier
train_step_class = tf.train.AdamOptimizer(learning_rate_class).minimize(loss_class)


#accuracy#
##########

#for classifier accuracy and evaluation
prediction = tf.argmax(y_class,1)
correct_prediction = tf.equal(tf.argmax(y_class,1), tf.argmax(y_class_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


##########
#training#
##########

#initialize#
############

#initialize session saver
saver = tf.train.Saver()

#create session object
session = tf.Session()

#initialize session
init = tf.initialize_all_variables()
session.run(init)

#for logging for autoencoder
losses = []

#for logging for classifier
losses_class = []
accuracies = []


#train autoencoder#
###################

if is_load_trained_autoencoder:
    #instead of training, load a saved session
    save_path = "autoencoder_trained.ckpt"
    saver.restore(session, save_path)
    print("Model restored from file: %s" % save_path)
else:
    start_time = time.time()
    
    #train
    for i in range(step_number):

        #create feed_dict for this step
        batches, labels_batch = next_batch(sentence_list, num_unrollings, batch_size = batch_size)
        feed_dict = dict()
        for j in range(num_unrollings + 1):
            feed_dict[x[j]] = batches[j]

        #train step
        session.run(train_step, feed_dict=feed_dict)

        #log loss and print ptogress
        if (i % log_step_number) == 0:
            #loss of the actual training sample
            loss_out = session.run(loss, feed_dict=feed_dict)
            losses.append(loss_out)

            #print progress
            print(str((float(i) / step_number)*100) + '%')

   
    training_time = time.time() - start_time
    print('Training time: ' + str(training_time))

    #plot loss
    plt.plot(losses)

#save session
if is_save_session:
    save_path = saver.save(session, "autoencoder_trained.ckpt")
    print("Model saved in file: %s" % save_path)
    
    
#train classifier#
##################

#function to calculate accuracy on multiple samples
def accuraccy_calc(embedded, num_unrollings, labes, batch_size):
    accuracy_sum = 0
    for index, _ in enumerate(embedded):

        #create feed_dict for this step
        batches, labels_batch = next_batch(embedded, num_unrollings, labels = labes, batch_size = batch_size, index = index)
        feed_dict = dict()
        for j in range(num_unrollings + 1):
            feed_dict[x[j]] = batches[j]
        feed_dict[y_class_] = labels_batch

        #calculate accuracy for this test step
        accuracy_out = session.run(accuracy, feed_dict=feed_dict)

        #sum of accuraccy for all the test samples
        accuracy_sum += accuracy_out

    accuraccy_average = accuracy_sum / float(len(embedded))
    
    return accuraccy_average


if is_load_trained_classifier:    
    save_path = "autoencoder_classifier_trained.ckpt"
    saver.restore(session, save_path)
    print("Model restored from file: %s" % save_path)
else:
    #train
    start_time = time.time()
    for i in range(step_number_class):

        #create feed_dict for this step
        batches, labels_batch = next_batch(training_embedded, num_unrollings, labels = training_labes, batch_size = batch_size)
        feed_dict = dict()
        for j in range(num_unrollings + 1):
            feed_dict[x[j]] = batches[j]
        feed_dict[y_class_] = labels_batch

        #train step
        session.run(train_step_class, feed_dict=feed_dict)

        #log loss and print ptogress
        if (i % log_step_number_class) == 0:
            #loss of the actual training sample
            loss_class_out = session.run(loss_class, feed_dict=feed_dict)
            losses_class.append(loss_class_out)

            #print progress
            print(str((float(i) / step_number_class)*100) + '%')


        #log accuracy
        if (i % log_step_number_accuracy) == 0:

            #calculate and append accuracy
            accuraccy_average = accuraccy_calc(test_embedded[:100], num_unrollings, test_labes[:100], batch_size)
            accuracies.append(accuraccy_average)
            print(accuraccy_average)



    training_time = time.time() - start_time
    print('Training time: ' + str(training_time))

    #plot loss
    plt.plot(losses_class)


#########################
#print training examples#
#########################

#define function to restore string from the embedded sentence
def sentence_embedded_to_string(embeddings, sentence_embedded):
    dist = np.dot(embeddings, np.transpose(sentence_embedded))
    word_indeces = np.argmax(dist, axis=0)
    sentence_string = ''
    for word_index in word_indeces:
        sentence_string += reverse_dictionary[word_index] + ' '
    return sentence_string


#get the first training sentence to the feed directory
batches, labels_batch = next_batch(sentence_list, num_unrollings, batch_size = batch_size, index = 0)
feed_dict = dict()
for j in range(num_unrollings + 1):
    feed_dict[x[j]] = batches[j]

#generate prediction
y_concat_out = session.run(y_concat, feed_dict=feed_dict)

#print results, using the embedding -> string function
print('Original sentence: ')
print(sentence_embedded_to_string(final_embeddings, sentence_list[0]))
print('Decoded sentence: ')
#the variable y_concat contains the prediction for all the batch elements, so the first batch element is selected
y_concat_out_first_element = y_concat_out[1::batch_size]
print(sentence_embedded_to_string(final_embeddings, y_concat_out_first_element))


#################################
#use model to generate sentences#
#################################

sentence = 'foo fighters watch video'


#cleanup sentence
#remove special characters
def remove_non_ascii_128(text):
    return ''.join([i if ord(i) < 128 else ' ' for i in text])
sentence = remove_non_ascii_128(sentence)

#remove some further undesirable characters
def remove_some_chars(text):
    chars_to_remove = set(['[', ']', '(', ')', "'", '"', '!', '?', '.', ',', '|', '@'])
    return ''.join([i for i in text if i not in chars_to_remove])
sentence = remove_some_chars(sentence)

#to lower case
sentence = sentence.lower() 


#sentence embedding
sentence_words = sentence.split()
sentence_length = len(sentence_words)
if sentence_length > (num_unrollings + 1):
    disp('Warning the sentence is too long, I will use the beginning  of it anyway.')

sentence_embedded = np.zeros((num_unrollings + 1, embedding_size))
for index_in_sentence in range(num_unrollings + 1):
    if index_in_sentence <= (sentence_length - 1):
        word = sentence_words[index_in_sentence]
        try:
            word_index = dictionary[word]
        except:
            word_index = 0
        sentence_embedded[index_in_sentence] = final_embeddings[word_index]
        
        
#get it to the right format
sentence_embedded_list = list()
sentence_embedded_list.append(sentence_embedded)
sentence_embedded_batch, labels_batch = next_batch(sentence_embedded_list, num_unrollings, batch_size = batch_size)


#create feed_dict
feed_dict = dict()
for j in range(num_unrollings + 1):
    feed_dict[x[j]] = sentence_embedded_batch[j]

#make preiction (outputs of decoder)
y_concat_out = session.run(y_concat, feed_dict=feed_dict)


print('Input sentence: ')
print(sentence_embedded_to_string(final_embeddings, sentence_embedded_list[0]))
print('Generated sentence: ')
#the variable y_concat contains the prediction for all the batch elements, so the first batch element is selected
y_concat_out_first_element = y_concat_out[1::batch_size]
print(sentence_embedded_to_string(final_embeddings, y_concat_out_first_element))


################################
#save states fore each sentence#
################################

sentence_list_length = len(sentence_list)

sentence_list_encoded = np.zeros((sentence_list_length, hidden_dim_2))
for index in range(sentence_list_length):
    #get the index-th training sentence to the feed directory
    batches, labels_batch = next_batch(sentence_list, num_unrollings, batch_size = batch_size, index = index)
    feed_dict = dict()
    for j in range(num_unrollings + 1):
        feed_dict[x[j]] = batches[j]

    #encode sentence to the lstm state
    code_out = session.run(code, feed_dict=feed_dict)
    #code_out is a butch, so get the first element of the batch (all the elements are the same)
    code_out = code_out[0]
    
    #store in the numpy array
    sentence_list_encoded[index,:] = code_out
    
    
with open('gigers_dataset_encoded_1000.pickle', 'wb') as f:  # Python 3: open(..., 'wb'), Python 3: open(..., 'w')
    pickle.dump(sentence_list_encoded, f)


########
#search#
########

sentence = 'jay z buys champagne brand'

#cleanup sentence
#remove special characters
def remove_non_ascii_128(text):
    return ''.join([i if ord(i) < 128 else ' ' for i in text])
sentence = remove_non_ascii_128(sentence)

#remove some further undesirable characters
def remove_some_chars(text):
    chars_to_remove = set(['[', ']', '(', ')', "'", '"', '!', '?', '.', ',', '|', '@'])
    return ''.join([i for i in text if i not in chars_to_remove])
sentence = remove_some_chars(sentence)

#to lower case
sentence = sentence.lower() 


#sentence embedding
sentence_words = sentence.split()
sentence_length = len(sentence_words)
if sentence_length > (num_unrollings + 1):
    disp('Warning the sentence is too long, I will use the beginning  of it anyway.')

sentence_embedded = np.zeros((num_unrollings + 1, embedding_size))
for index_in_sentence in range(num_unrollings + 1):
    if index_in_sentence <= (sentence_length - 1):
        word = sentence_words[index_in_sentence]
        try:
            word_index = dictionary[word]
        except:
            word_index = 0
        sentence_embedded[index_in_sentence] = final_embeddings[word_index]
        
        
#get it to the right format
sentence_embedded_list = list()
sentence_embedded_list.append(sentence_embedded)
sentence_embedded_batch, labels_batch = next_batch(sentence_embedded_list, num_unrollings, batch_size = batch_size)


#create feed_dict
feed_dict = dict()
for j in range(num_unrollings + 1):
    feed_dict[x[j]] = sentence_embedded_batch[j]

#encode sentence to the code (most dense data)
code_out = session.run(code, feed_dict=feed_dict)
code_out = code_out[0]


#define function to search closest codes to the imput code, and output them as string
def search_closest(encodings_list, encoding, sentence_list_string):
    dist = np.dot(encodings_list, np.transpose(encoding))
    top_k = 8 # number of nearest codes
    nearest_indeces = (-dist).argsort(axis=0)[0:top_k-1]
    #closest_index = np.argmax(dist, axis=0)
    search_results = [sentence_list_string[index] for index in nearest_indeces]
    return search_results

#search the closest sentences in all the sentences, to the search expression
search_results = search_closest(sentence_list_encoded, code_out, sentence_list_string)

print('Search expression:')
print(sentence)
print('Search results:')
print(search_results)


#####################
#test classification#
#####################

#classify the unlabelled gigers news titles
predictions = []
for index, sentence in enumerate(sentence_list_embedded):
    
    batches, labels_batch = next_batch(sentence_list_embedded, num_unrollings, batch_size = batch_size, index = index)
    feed_dict = dict()
    for j in range(num_unrollings + 1):
        feed_dict[x[j]] = batches[j]
        
    prediction_out = session.run(prediction, feed_dict = feed_dict)
    prediction_out = prediction_out[0] #all the elements of the batch is the same
    predictions.append(prediction_out)
    

#print news predicted as new music
for index, sentence in enumerate(sentence_list_string):
    if predictions[index] == 1:
        print(sentence)


