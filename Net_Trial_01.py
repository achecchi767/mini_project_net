from mnist_import_chat import mnist_import
training_data, validation_data, test_data = mnist_import()
import gc


import Network_00
net = Network_00.Network([784, 30, 10]) # the three parameters set here are the number of input (784), hidden (30), and output layer (10)

net.SGD(training_data, 15, 10, 3.0, test_data=test_data) 
# this is the training step, the first number parameter is the number of epoch, followed by batch size and learning rate
camp = False
if camp:# switch to True when I finish my session to clean from data imported
    del training_data, validation_data, test_data, net
    gc.collect()

# The general idea here is to identify the set of weights from hidden to output layer that influence the most the activation of an output neuron
# This is achieved with a simple procedure, take the weights, normalize them on a scale -1, 1 then choose only the one > 0.5
# Now we have identified the neurons of the hidden layer, which activation should influence the most the final output of the net on a specific digit example
# To see on what aspects of the input these "relevant decision neurons" (RDN) are picking on, I plot for each of them the weights from input to hidden layer
# These weights when plotted (blu for positive, red for negative) should give an intuition of the patterns these RDN were selective for

run = True # switch to false if you want to train the network without visualizing any neuron
if run:
    weights = net.weights # takes all the weights in the network 
    for output_n in range(2): 
        list_weights_neuron = weights[1][output_n] # at each cycle select only the weights from hidden to output, linked to one output neuron
        
        from visualize_net import normalize_to_range_weight, visualize_neuron
        list_weights_neuron_N = normalize_to_range_weight(list_weights_neuron) # normalized on a scale from -1 to 1
        index = -1
        relevant_neurons = []
        for n_h in list_weights_neuron_N: # for each connection
            index += 1 
            if n_h > 0.5: # arbitrary rule based on weights observation (only few above 0.5 usually): check and append only the important one
                relevant_neurons.append(index)

        input_weights_relevant_neuron = []
        for r_n in relevant_neurons:
            input_weights_relevant_neuron.append(weights[0][r_n])
        # here I am creating a list with as items, the list of weights (input-hidden) for the relevant neurons 
        print(len(input_weights_relevant_neuron))
             
        for r_v in input_weights_relevant_neuron:
            visualize_neuron(r_v)
        # here I show, in sequence, each map\image for each relevant neuron of a specific digit