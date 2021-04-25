import { Perceptron } from "./perceptron";
import { Sigmoid } from "../activateMethods/activationMethods";

export class NeuralMLPNetwork {
  /**
   * Total epochs to train
   */
  private epochs;

  private learningRate: number;

  private hiddenLayers: Perceptron[][];

  constructor(
    epochs: number,
    learningRate: number,
    quantityHiddenLayers: number,
    quantityInputs: number,
    quantityOutput: number
  ) {
    this.epochs = epochs;
    this.learningRate = learningRate;
    this.hiddenLayers = [];

    this.hiddenLayers.push(
      ...Array(quantityHiddenLayers)
        .fill(0)
        .map((_, layerPos) => {
          const layerPercptronInputs =
            layerPos > 0 ? quantityInputs + 1 : quantityInputs;
          return Array(quantityInputs + 1)
            .fill(0)
            .map(() => new Perceptron(layerPercptronInputs, new Sigmoid()));
        })
    );

    // add to layers last perceptron
    this.hiddenLayers.push(
      ...Array(quantityOutput)
        .fill(0)
        .map(() =>
          Array(quantityOutput)
            .fill(0)
            .map(() => new Perceptron(quantityInputs + 1, new Sigmoid()))
        )
    );
  }

  private backwardPropagateError(expected: number, result: number) {
    const err = expected - result;
    [...this.hiddenLayers].reverse().forEach((layer, indexLayer, clone) => {
      if (!indexLayer) {
        layer.forEach((neuron) => {
          neuron.calcDeltaOutPerceptron(err, result);
        });
      } else {
        layer.forEach((neuron, index) => {
          clone[indexLayer - 1].forEach((next) => {
            let deltaOutput = next.getDelta();
            let weight = next.getWeights()[index];
            neuron.calcDeltaHiddenLayer(deltaOutput, weight);
          });
        });
      }
    });
  }

  /**
   * execute a forward propagation with one sample
   * @param sample input sample
   * @returns result of sums
   */
  private forwardPropagate(sample: number[]): number[] {
    let inputs = [...sample];
    this.hiddenLayers.forEach((layer) => {
      let newInputs: number[] = [];
      layer.forEach((neuron) => {
        let neuronOutput = neuron.calc(inputs);
        newInputs.push(neuronOutput);
      });
      inputs = [...newInputs];
    });

    return [...inputs];
  }

  private updateWeights(inputs: number[]) {
    this.hiddenLayers.forEach((layer, indexLayer) => {
      let auxInputs: number[] = [...inputs];
      if (indexLayer !== 0) {
        // is isn't the first layer, gets previous output
        auxInputs = this.hiddenLayers[indexLayer - 1].map((neuron) =>
          neuron.getOutput()
        );
      }

      layer.forEach((neuron) => {
        auxInputs.forEach((input, indexInput) => {
          const currentWeight = neuron.getWeights()[indexInput];
          const newWeight =
            currentWeight * 1 + input * neuron.getDelta() * this.learningRate;
          neuron.setWeight(newWeight, indexInput);
        });
      });
    });
  }

  public predict(input: number[]): number {
    const [result] = this.forwardPropagate(input);
    return result;
  }
  /**
   * Execute the train of the model
   * @param inputs matriz of inputs
   * @param targets respective targets of the inputs
   * @returns {number} acurracy
   */
  public train(inputs: number[][], targets: number[]): number {
    let countEpoch = 0;
    let erros = [] as number[];

    while (countEpoch < this.epochs) {
      erros = [];
      inputs.forEach((input, index) => {
        const target = targets[index];
        const [result] = this.forwardPropagate(input);
        const err = target - result;
        if (target !== result) {
          this.backwardPropagateError(target, result);
          this.updateWeights(input);
        }
        erros.push(Math.abs(err));
      });
      countEpoch++;
    }

    const mean = erros.reduce((a, b) => a + b, 0) / erros.length;
    return Math.abs(mean - 1);
  }
}
