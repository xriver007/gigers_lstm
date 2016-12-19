
import pickle
import numpy as np


################
#load embedding#
################

#Load data (embeddings, dictionary and reverse dictionary)
with open('gigers_dataset_dictionary_embedding.pickle') as f:  # Python 3: open(..., 'rb')
    final_embeddings, dictionary, reverse_dictionary = pickle.load(f)

dictionary_size, embedding_size = final_embeddings.shape

#test embeddings
print(dictionary['the'])
print(reverse_dictionary[0])


############################
#load and process text data#
############################

#load text as data
with open ("gigers_dataset.txt", "r") as datat_text_file:
    data_text = datat_text_file.readlines()
    datat_text_file.close()
data_text = data_text[0]   

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
    
#separate to sentences
sentence_list_string = data_text.split('<titleseparator>')


#################
#embed sentences#
#################

#we will use the subset of the dataset
sentence_number = 100
sentence_list_string = sentence_list_string[:sentence_number]

#split sentences to words
sentence_words_list_string = [ sentence.split() for sentence in sentence_list_string ]

#a set of equally long sequences will be needed
max_sentence_length = max(len(sentence) for sentence in sentence_words_list_string)

#make the embedding sentence by senence
sentence_list_embedded = []
for sentence_string in sentence_words_list_string:
    sentence_embedded = np.zeros((max_sentence_length, embedding_size))
    for index_in_sentence, word in enumerate(sentence_string):
        try:
            word_index = dictionary[word]
        except:
            word_index = 0
        sentence_embedded[index_in_sentence] = final_embeddings[word_index]
    sentence_list_embedded.append(sentence_embedded)
    

######
#save#
######
save_pickle = True

if save_pickle:
    # Saving the objects:
    with open('gigers_dataset_embedded.pickle', 'wb') as f:  # Python 3: open(..., 'wb'), Python 3: open(..., 'w')
        pickle.dump([sentence_list_embedded, sentence_list_string], f)
        
print('gigers_dataset_embedded saved')
