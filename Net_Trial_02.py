
from mnist_import_chat import mnist_import

import gc
from visualize_net import normalize_to_range_weight, visualize_neuron, normalize_to_range_activations
import numpy as np
import Network_00

training_data, validation_data, test_data = mnist_import()
net = Network_00.Network([784, 30, 10])
net.SGD(training_data, 15, 10, 3.0, test_data=test_data)

camp = True
if camp:# switch to True when I finish my session
    del training_data, validation_data, test_data, net
    gc.collect()

# The assumption is that some neurons in the hidden layer are more influencial then others when it come to the final decision 
# for the classification of a specific digit. Units in the hidden layer that are more active for a certain input should be 
# the more influencial. So the first step is to find this neurons. To do so I feed an input to the trained network and I record 
# the top 5 neurons out of the 30 in the hidden layer. Then I look at the weights each of these neurons received from the input layer
# I average all five to look for an emerging pattern, that should intuitivelt come out (a circle, a diagonal line, ...) 

run = True # to not run the map change to False
for i in range(1):
    activations = net.feedforward_keep_trial(training_data[i][0]) # collect all activations from an example trial 
    activations_hidden = activations[0] # take only activation from the hidden layer
    activations_hidden_N = normalize_to_range_activations(activations_hidden) # normalize them on a 0 to 1 scale
 
    counter = -1
    top_ind = []
        
    for r_n in activations_hidden_N: # pick one activation value
        counter += 1
        if len(top_ind) == 5: #when we have the top five exit the loop
            break
        for r_i in activations_hidden_N: #compare it with all the others
            if r_n >= r_i and counter not in top_ind: #find the new winner
                top_ind.append(counter) #add it to the list
    

    run = True # true to visualize, false to skip
    if run:
        
        weights = net.weights #access all weights
        input_weights_relevant_neuron = []
        for r_w in top_ind:
            input_weights_relevant_neuron.append(weights[0][r_w]) 
            # here I am creating a list with as items, the list of weights (input-hidden) for the relevant neurons (top5)
        final_image = np.zeros_like(weights[0][0]) #initializing the final image vector
        input_weights_relevant_neuron_N = normalize_to_range_weight(input_weights_relevant_neuron) # normalize to a -1, 1 scale
        final_image = np.mean(input_weights_relevant_neuron, axis=0) # take the mean among the top five
            
        visualize_neuron(final_image) 
       
        
        
       