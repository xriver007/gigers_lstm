#based on: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/word2vec/word2vec_basic.py

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pickle
import numpy as np
import colorsys


###########
#load data#
###########

#load encoded text data
with open('gigers_dataset_encoded_1000.pickle') as f:  # Python 3: open(..., 'rb')
    sentence_list_encoded = pickle.load(f)


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
 
    
#separate to sentences
sentence_list_string = data_text.split('<titleSeparator>')


#Load k means assignments
with open('gigers_dataset_kmeans_assignments_1000.pickle') as f:  # Python 3: open(..., 'rb')
    assignments = pickle.load(f)


################
#visualize data#
################

def plot_with_labels(low_dim_embs, labels, assignments, rgb_color_palette, is_annotate_labels = True, filename='gigers_dataset_encoded_1000_visualized.png'):
  assert low_dim_embs.shape[0] >= len(labels), "more labels than embeddings"
  plt.figure(figsize=(18, 18))  # in inches
  for i, label in enumerate(labels):
    x, y = low_dim_embs[i, :]
    plt.scatter(x, y, color=rgb_color_palette[assignments[i]])
    if is_annotate_labels:
        plt.annotate(label,
                    xy=(x, y),
                    xytext=(5, 2),
                    textcoords='offset points',
                    ha='right',
                    va='bottom').set_alpha(.4)

  plt.savefig(filename)
  plt.show()


#generate color palette, for the different assignments (calsses)
color_numer = assignments.max() + 1
hsv_color_palette = [(x*1.0/color_numer, 1, 1) for x in range(color_numer)]
rgb_color_palette = map(lambda x: colorsys.hsv_to_rgb(*x), hsv_color_palette)
  
#create TSNE
tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
plot_only = 1000
low_dim_embs = tsne.fit_transform(sentence_list_encoded[:plot_only, :])
labels = [sentence_list_string[i] for i in xrange(plot_only)]

#plot
plot_with_labels(low_dim_embs, labels, assignments, rgb_color_palette, is_annotate_labels = False)