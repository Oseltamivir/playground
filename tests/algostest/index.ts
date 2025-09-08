import { diffWeights, addScaled, zerosLike, aggregateDeltas} from "../../src/fl/algorithms";
import { Weights } from "../../src/fl/types";
import { test, run, assertArrayClose } from "../_util/harness";

function W(arrs: number[][]): Weights { return arrs.map(a => new Float32Array(a)); }

// Test 1: Simple element-wise difference
test("diffWeights", () => {
  const a = W([[1, 2], [3]]);
  const b = W([[0, 1], [10]]);
  const d = diffWeights(a, b);
  assertArrayClose(d[0], new Float32Array([1, 1]));
  assertArrayClose(d[1], new Float32Array([-7]));
});

// Test 2: Elementwise a + s·b.
test("addScaled", () => {
  const a = W([[1, 2]]);
  const b = W([[3, -1]]);
  const out = addScaled(a, b, 0.5);
  assertArrayClose(out[0], new Float32Array([1 + 0.5 * 3, 2 + 0.5 * -1]));
});

// Test 3: torch.zeros_like
test("zerosLike", () => {
  const z = zerosLike(W([[5, 6, 7]]));
  assertArrayClose(z[0], new Float32Array([0, 0, 0]));
});

// Test 4: Returns a plain average of client deltas
test("aggregateDeltas unweighted avg", () => {
  const deltas = [
    { delta: W([[1, 2]]), weight: 10 },
    { delta: W([[3, 4]]), weight: 20 }
  ];
  const out = aggregateDeltas(deltas as any, false);
  assertArrayClose(out[0], new Float32Array([2, 3])); // average of rows
});

// Test 5: Returns a weighted average of client deltas
test("aggregateDeltas weighted", () => {
  const deltas = [
    { delta: W([[1, 2]]), weight: 1 },
    { delta: W([[3, 4]]), weight: 3 }
  ];
  const out = aggregateDeltas(deltas as any, true);
  // (1*[1,2] + 3*[3,4]) / (1+3) = [ (1+9)/4, (2+12)/4 ] = [2.5, 3.5]
  assertArrayClose(out[0], new Float32Array([2.5, 3.5]));
});


run();