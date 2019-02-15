Training BiDAF model using SQuaD dataset
This is my implementation of BiDAF model from scratch in pytorch. 

 Structure of the code
  - Import relevant packages
  - Load the input dataset 
      - SQuAD
      - GLoVe 
  - Create a Bidirectional LSTM with attention model in pytorch
  - Create vectors for context and query inputs
      - check for grammar errors and incorrect words
      - how will I account for grammatical errors?
          - is there a simple grammatical check that I can run on the text?
              - Spelling correction using textblob (nad accuracy - use character level emebeddings)
  - create the set of questions that are to be queried
  - train the model on SQuAD dataset

Training tips:
- ensure you convert all the alphabets to their lower case before looking them up in the Glove model embeddings
- ensure you convert the phonetic symbols to alphabets for better embedding represnetation
- cleaning data is very very important
	- special characters
	- did not deal wiht answers which were only special characters
	- cleaned out all punctuations -- OOV for 'child.' so cleaning is important

- monitoring the GPU Utilization : command line instruction ---->  nvidia-smi
- use AdamOptimizer
- saving and loading the model
	-saving all the optimizer values and the loss variable
- optimizing individual steps
	- optmizing the call to the model dictionary for evey paragraph
- call for flatten_parameters()	before training the network with LSTM/RNN
- if segmentation fault occurs, try decreasing the hidden size, batch size, 
- if the NN is not converging
	- modify the learning rate
	- try overfitting with smaller training sample size
	- check for autograd/ zero grad variables
- do not use an activation function at the output layer 
- add bias in all the layers!
- while training on multiple gpus use - loss.sum().backward()
- check if the GPUs are acceessed by pytorch - print(torch.cuda.is_available())

Other GREAT references:
Tips for training by Andrej Karpathy: https://twitter.com/karpathy/status/1013244313327681536?s=19
https://blog.slavv.com/37-reasons-why-your-neural-network-is-not-working-4020854bd607
Difference between detach, no_gra, requires_grad
https://discuss.pytorch.org/t/detach-no-grad-and-requires-grad/16915
Difference bwteen .view() and .permute()

