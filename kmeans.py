#based on: https://gist.github.com/dave-andersen/265e68a5e879b5540ebc

import tensorflow as tf
import numpy as np
import time
import pickle


#load encoded text data
with open('gigers_dataset_encoded_1000.pickle') as f:  # Python 3: open(..., 'rb')
    sentence_list_encoded = pickle.load(f)


#########
#k means#    
#########
	
#sentence_list_encoded is shaped as (number_of_sentences, cell_size)
dimensios = sentence_list_encoded.shape[1]

N=sentence_list_encoded.shape[0]
K=100
MAX_ITERS = 1000
is_cosine_distance = True

start = time.time()

points = tf.Variable(sentence_list_encoded)
cluster_assignments = tf.Variable(tf.zeros([N], dtype=tf.int64))

#silly initialization:  use the first K points as the starting centroids.
centroids = tf.Variable(tf.slice(points.initialized_value(), [0,0], [K,dimensios]))

#replicate to N copies of each centroid and K copies of each point, then subtract and compute the sum of squared distances.
rep_centroids = tf.reshape(tf.tile(centroids, [N, 1]), [N, K, dimensios])
rep_points = tf.reshape(tf.tile(points, [1, K]), [N, K, dimensios])
sum_squares = tf.reduce_sum(tf.square(rep_points - rep_centroids), 
                            reduction_indices=2)

#cosine distances of points from centroids
distances_cosine = tf.matmul(points, centroids, transpose_b = True)

if is_cosine_distance:
    #the two vectors are the closest if distances_cosine is max, hence if -distances_cosine is min
    distances = -distances_cosine
else:
    distances = sum_squares

#use argmin to select the lowest-distance point
best_centroids = tf.argmin(distances, 1)
did_assignments_change = tf.reduce_any(tf.not_equal(best_centroids, 
                                                    cluster_assignments))

def bucket_mean(data, bucket_ids, num_buckets):
    total = tf.unsorted_segment_sum(data, bucket_ids, num_buckets)
    count = tf.unsorted_segment_sum(tf.ones_like(data), bucket_ids, num_buckets)
    return total / count

means = bucket_mean(points, best_centroids, K)

#do not write to the assigned clusters variable until after computing whether the assignments have changed - hence with_dependencies
with tf.control_dependencies([did_assignments_change]):
    do_updates = tf.group(
        centroids.assign(means),
        cluster_assignments.assign(best_centroids))

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

changed = True
iters = 0

while changed and iters < MAX_ITERS:
    iters += 1
    [changed, _] = sess.run([did_assignments_change, do_updates])

[centers, assignments] = sess.run([centroids, cluster_assignments])
end = time.time()

print(("Found in %.2f seconds" % (end-start)), iters, "iterations")
print("Centroids:")
print(centers)
print("Cluster assignments:", assignments)

assignment_counts = np.bincount(assignments)
print("Number of elements in each class")
print(assignment_counts)

assignment_sorted = (-assignment_counts).argsort()
print("Assignments in descending order")
print(assignment_sorted)

with open('gigers_dataset_kmeans_assignments_1000.pickle', 'wb') as f:  # Python 3: open(..., 'wb'), Python 3: open(..., 'w')
    pickle.dump(assignments, f)

	
######################
#plot news of a class#    
######################

#class to print
class_selected = assignment_sorted[0] #the class with the most element

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

#the first N sentence is used for now
sentence_list_string = sentence_list_string[:N]

#select the sentence_list_string indeces, with the given class
class_indeces = np.where(assignments == class_selected)[0]

#select the sentences with the given class
class_sentences = [sentence_list_string[x] for x in class_indeces]

#print the results
print(class_sentences)