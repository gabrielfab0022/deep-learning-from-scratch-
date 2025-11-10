# deep-learning-from-scratch-
Este repositório implementa redes neurais (MLP, CNN, Autoencoders) totalmente do zero, sem PyTorch / TensorFlow / Keras. Todas as camadas, forward, backward, loop de treino e otimizadores foram programados manualmente.

Dei o nome de fran_torch porque meu nome é Francisco :) uma versão minimalista / caseira / educacional do PyTorch, porém implementada inteiramente à mão, para estudo e aprendizado profundo dos fundamentos (aprendizado profundo meu, no caso, e não do modelo haha).

Para usar o fran_torch basta definir um modelo como uma lista de structs, em que cada elemento da lista é uma camada da rede. No fran_torch, existem apenas 6 tipos de camadas: 
conv2d (forward de uma camada de convolução 2d)
maxpooling (camada de max pooling)
flatten (converte um sinal 2d, com ---possivelmente --- múltiplos canais em um vetor)
dense (camada densa)
conv2d_transpose (camada de convolução transposta)
softmax (camada de softmax)

Para criar cada uma dessas camadas, basta definir uma struct e passar como parâmetro os atributos da camada. Por exemplo, para a convolução 2d eu preciso definir o tamanho dos kernels, a quantidade de kernels, o stride e a função de ativação. Para uma camada densa eu preciso definir a quantidade neurônios na saída e a função de ativação... Bom, melhor mostrando do que tá falando... aqui vai a definição de um modelo pequeno, inspirado na LeNet5: 

model = network([conv2d(k=5, channels=6, stride=1, activation="tanh"),
                 maxpooling(k=2, stride=2),
                 conv2d(k=3, channels=16, stride=1, activation="tanh"),
                 maxpooling(k=2, stride=2),
                 flatten(),
                 dense(n_out=120, activation="tanh"),
                 dense(n_out=84, activation="tanh"),
                 dense(n_out=10, activation="tanh"),
                 softmax() 
], 0, 0)

model = init_weights(model, m_in, n_in);

viu como é fácil? kkk 
Para fazer uma época de treinamento, basta usar a função train_nn, e para fazer o treinamento completo basta colocar dentro de um loop. 
Para avaliar o modelo em um dado conjunto de dados, basta usar a função evaluate_nn.

No repositório, vou colocar também alguns exemplos em que eu uso a fran_torch para algumas tarefas simples: treinamento da LeNet5 com 10000 amostras da mnist, e avaliação na base de teste, treinamento de uma outra rede convolutiva para classificação da fmnist, também com um subconjunto de 10000 amostras para treino, e treinamento de um autoencoder convolutivo. 



English description: 
This repository implements neural networks (MLP, CNN, Autoencoders) completely from scratch, without PyTorch / TensorFlow / Keras. All layers, forward pass, backward pass, training loop and optimizers were manually programmed.

I named this little library fran_torch because my name is Francisco :) it is a minimalistic / homemade / educational PyTorch-style version, but implemented entirely by hand, for studying and deeply understanding the fundamentals (deep learning for me, in this case, not for the model haha).

To use fran_torch, you just need to define a model as a list of structs, where each element of the list is a layer of the network. In fran_torch, there are only 6 types of layers:
conv2d (forward of a 2D convolution layer)
maxpooling (max pooling layer)
flatten (converts a 2D signal, with — possibly — multiple channels, into a vector)
dense (fully connected layer)
conv2d_transpose (transposed convolution layer)
softmax (softmax layer)

To create each of these layers, you just define a struct and pass the attributes of the layer as parameters. For example, for a 2D convolution I need to define the kernel size, the number of kernels, the stride and the activation function. For a dense layer I need to define the number of output neurons and the activation function… well, easier showing than explaining… here is the definition of a small model inspired by LeNet5:

model = network([conv2d(k=5, channels=6, stride=1, activation="tanh"),
                 maxpooling(k=2, stride=2),
                 conv2d(k=3, channels=16, stride=1, activation="tanh"),
                 maxpooling(k=2, stride=2),
                 flatten(),
                 dense(n_out=120, activation="tanh"),
                 dense(n_out=84, activation="tanh"),
                 dense(n_out=10, activation="tanh"),
                 softmax() 
], 0, 0)

model = init_weights(model, m_in, n_in);


see how easy it is? haha
To perform one training epoch, just use the function train_nn, and to perform full training just put it inside a loop.
To evaluate the model on a given dataset, just use the function evaluate_nn.

In the repository, I will also include some examples where I use fran_torch for some simple tasks: training LeNet5 with 10,000 MNIST samples (and evaluating on the test set), training another convolutional network for FMNIST classification also with a 10,000-sample training subset, and training a convolutional autoencoder.
