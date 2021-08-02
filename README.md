## Implementação de perceptron multicamadas (do ingles multilayer perceptron)

Neste repositório foi implementado com código typescript puro um MLP, capaz de solucionar problemas não lineares de forma consistente.

### Perceptron

O perceptron consiste em um algoritmo de aprendizado com classifação binária, desta forma ele é capaz de solucionar apenas problemas linearmente separáveis.

### MLP 

O perceptron multicamadas é uma rede neural semelhante à perceptron, mas com mais de uma camada de neurônios em alimentação direta. 
Tal tipo de rede é composta por camadas de neurônios ligadas entre si por sinapses com pesos
#### Retropropagação (Backpropagation)

Este código faz uso do algoritmo de retro-propagação para garantir que os perceptrons possam aprender a medida que as "épocas" passem.

Os pesos são iniciados aleatoriamente em um intervalo de 0 até 1. A medida que as épocas passem sobre o dataset cujo qual está sendo treinado, os pesos se ajustam permitindo que as classificações se aproximem do target e assim o algoritmo aprenda.

#### Camadas

O algoritmo implementado permite que o utilizador faça uso de quantas camadas ele preferir. Porém o código em si implementa 3 (três) camadas de perceptrons.
