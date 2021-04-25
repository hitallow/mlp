import { ActivateFnContract } from "../contracts/activateFn";

export class Sigmoid implements ActivateFnContract {
  public derivate(v: number): number {
    return v * (1.0 - v);
  }
  public activate(activation: number): number {
    return 1 / (1 + Math.exp(-activation));
  }
}
