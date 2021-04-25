import { NeuralMLPNetwork } from "./models/mlp";

const epocas = 1000000;
const learningRate = 0.9;
const hiddenLayers = 1;
const quantityInputs = 2;
const quantityOutputs = 1;

const mlp = new NeuralMLPNetwork(
  epocas,
  learningRate,
  hiddenLayers,
  quantityInputs,
  quantityOutputs
);

console.log("Quantidade de epocas", epocas);

const acurracy = mlp.train(
  [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
  ],
  [0, 1, 1, 0]
);
console.log("Acuracia", acurracy);

console.log("Predict 0 - 0  = ", mlp.predict([0, 1]));
