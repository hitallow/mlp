/**
 * Create the weights randomly
 * @param quantity total weights quantity
 */
export function startWeightsRandomly(quantity: number): number[] {
  return Array(quantity)
    .fill(0)
    .map(() => Math.random());
}
