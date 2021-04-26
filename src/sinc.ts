import { NeuralMLPNetwork } from "./models/mlp";
console.group("---------------- SINC PROBLEM ----------------");

const epocas = 111000;
const learningRate = 0.3;
const hiddenLayers = 1;
const quantityInputs = 1;
const quantityOutputs = 1;

function sinc(x: number): number {
  if (x === 0) return 1;
  return Math.sin(Math.PI * x) / (Math.PI * x);
}

function generatePoints(total: number): [number[][], number[]] {
  let inputs: number[][] = [];
  let targets: number[] = [];

  for (let i = -total; i < total; i++) {
    if (i === 0) continue;
    inputs.push([i]);
    targets.push(sinc(i));
  }

  return [inputs, targets];
}

const totalPoints = 10;

const [inputs, targets] = generatePoints(totalPoints);

const mlpSinc = new NeuralMLPNetwork(
  epocas,
  learningRate,
  hiddenLayers,
  quantityInputs,
  quantityOutputs
);

console.log("Start training SINC");

let acurracy = mlpSinc.train(inputs, targets);

console.log("End training SINC");

console.log("Sinc acurracy", acurracy);
const table = [];
for (let i = 1; i <= 100; i += 9) {
  const x1 = i;
  const expected = sinc(i);
  table.push({
    x1,
    expected,
    result: mlpSinc.predict([x1]).toExponential()
  });
}

console.table(table);

console.groupEnd();
