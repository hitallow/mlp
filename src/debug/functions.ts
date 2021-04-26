import { Perceptron } from "../models/perceptron";

export function printLayers(network: Perceptron[][]) {
  network.forEach((layer, i) => {
    console.group("Layer", i + 1);
    layer.forEach((x) => console.log(x));
    console.groupEnd();
  });
}
