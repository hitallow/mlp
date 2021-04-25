import { ActivateFnContract } from "../contracts/activateFn";
import { startWeightsRandomly } from "../helpers/helpers";

export class Perceptron {
  private weights: number[] = [];

  private bias: number = 0;

  private output: number = 0;

  private activateFunction: ActivateFnContract;

  private inputQuantity: number = 0;

  private delta = 0;

  constructor(inputQuantity: number, activateContract: ActivateFnContract) {
    this.inputQuantity = inputQuantity;
    this.activateFunction = activateContract;
    this.weights = startWeightsRandomly(inputQuantity);
    this.bias = Math.random();
  }

  public getWeights(): number[] {
    return [...this.weights];
  }

  public setWeight(weight: number, position: number): Perceptron {
    this.weights[position] = weight;
    return this;
  }

  public calcDeltaHiddenLayer(
    nextLayerDelta: number,
    nextLayerAssociatedWeight: number
  ) {
    this.delta =
      nextLayerDelta *
      nextLayerAssociatedWeight *
      this.activateFunction.derivate(this.output);

    return this;
  }

  public calcDeltaOutPerceptron(err: number, result: number) {
    this.delta = err * this.activateFunction.derivate(result);
    return this;
  }

  public getDelta(): number {
    return this.delta;
  }

  public addWeight(weight: number): Perceptron {
    this.weights.push(weight);
    return this;
  }

  public setWeights(weights: number[]): Perceptron {
    this.weights = [...weights];
    return this;
  }

  public getOutput(): number {
    return this.output;
  }

  public setActivateFunction(activate: ActivateFnContract): Perceptron {
    this.activateFunction = activate;
    return this;
  }

  /**
   * Active perceptron
   * @param input
   * @returns
   */
  public calc(input: number[]): number {
    let sum = 0;

    for (let i = 0; i < this.inputQuantity; i++) {
      sum += input[i] * this.weights[i];
    }
    sum += this.bias;

    const output = this.activateFunction.activate(sum);

    this.output = output;

    return this.output;
  }
}
