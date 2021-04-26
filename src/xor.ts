import { NeuralMLPNetwork } from "./models/mlp";

console.group("---------------- XOR PROBLEM ----------------");

const epocas = 100000;
const learningRate = 0.5;
const hiddenLayers = 2;
const quantityInputs = 2;
const quantityOutputs = 1;

const mlp = new NeuralMLPNetwork(
  epocas,
  learningRate,
  hiddenLayers,
  quantityInputs,
  quantityOutputs
);

console.log("Quantidade de épocas", epocas);

console.log("Start training XOR");
let acurracy = mlp.train(
  [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
  ],
  [0, 1, 1, 0]
);
console.log("End training XOR");
console.log("Acurácia", acurracy);

const predict = [
  {
    x1: 0,
    x2: 0,
    expected: 0,
    result: 0
  },
  {
    x1: 0,
    x2: 1,
    expected: 1,
    result: 0
  },
  {
    x1: 1,
    x2: 0,
    expected: 1,
    result: 0
  },
  {
    x1: 1,
    x2: 1,
    expected: 0,
    result: 0
  }
];

predict.forEach((v) => {
  v.result = Math.round(mlp.predict([v.x1, v.x2]));
});

console.table(predict);

console.groupEnd();
