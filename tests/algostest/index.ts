import { diffWeights, addScaled, zerosLike, aggregateDeltas, applyProximalGradient } from "../../src/fl/algorithms";
import { Weights } from "../../src/fl/types";
import { test, run, assertArrayClose } from "../_util/harness";

function W(arrs: number[][]): Weights { return arrs.map(a => new Float32Array(a)); }

test("diffWeights", () => {
  const a = W([[1, 2], [3]]);
  const b = W([[0, 1], [10]]);
  const d = diffWeights(a, b);
  assertArrayClose(d[0], new Float32Array([1, 1]));
  assertArrayClose(d[1], new Float32Array([-7]));
});

test("addScaled", () => {
  const a = W([[1, 2]]);
  const b = W([[3, -1]]);
  const out = addScaled(a, b, 0.5);
  assertArrayClose(out[0], new Float32Array([1 + 0.5 * 3, 2 + 0.5 * -1]));
});

test("zerosLike", () => {
  const z = zerosLike(W([[5, 6, 7]]));
  assertArrayClose(z[0], new Float32Array([0, 0, 0]));
});

test("aggregateDeltas unweighted avg", () => {
  const deltas = [
    { delta: W([[1, 2]]), weight: 10 },
    { delta: W([[3, 4]]), weight: 20 }
  ];
  const out = aggregateDeltas(deltas as any, false);
  assertArrayClose(out[0], new Float32Array([2, 3])); // average of rows
});

test("aggregateDeltas weighted", () => {
  const deltas = [
    { delta: W([[1, 2]]), weight: 1 },
    { delta: W([[3, 4]]), weight: 3 }
  ];
  const out = aggregateDeltas(deltas as any, true);
  // (1*[1,2] + 3*[3,4]) / (1+3) = [ (1+9)/4, (2+12)/4 ] = [2.5, 3.5]
  assertArrayClose(out[0], new Float32Array([2.5, 3.5]));
});

test("applyProximalGradient basic sanity", () => {
  const w  = W([[ 2,  0]]);
  const w0 = W([[ 1, -1]]);
  const d  = W([[ 0,  0]]);
  const out = applyProximalGradient(d, w, w0, 0.5, 1);
  // subtract (mu/steps)*(w - w0) = 0.5*[1, 1]
  assertArrayClose(out[0], new Float32Array([-0.5, -0.5]));
});

run();