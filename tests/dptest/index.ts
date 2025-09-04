import { clipUpdate, addGaussianNoise } from "../../src/fl/dp";
import { Weights } from "../../src/fl/types";
import { test, run, assertArrayClose } from "../_util/harness";

function W(arrs: number[][]): Weights { return arrs.map(a => new Float32Array(a)); }

// Test 1: clipUpdate
test("clipUpdate scales to cap", () => {
  const d = W([[3, 4]]); // norm = 5
  const out = clipUpdate(d, 2);
  assertArrayClose(out[0], new Float32Array([1.2, 1.6])); // should scale down to norm 2
});


// Deterministic Box–Muller by patching Math.random to a fixed sequence.
// dp.ts uses: u=Math.random(), v=Math.random(), n = sqrt(-2 ln u) * cos(2π v).
test("addGaussianNoise: deterministic via patched RNG", () => {
  const seq = [0.13579, 0.24680, 0.97531, 0.86420, 0.11111, 0.22222]; // >0
  let idx = 0;
  const orig = Math.random;
  (Math as any).random = function() { const v = seq[idx % seq.length]; idx++; return v; };

  const base = W([[1.0, -2.0, 0.5]]);
  const std = 0.1;

  // compute expected manually with the same sequence
  function z(u: number, v: number): number {
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  const z0 = z(seq[0], seq[1]), z1 = z(seq[2], seq[3]), z2 = z(seq[4], seq[5]);
  const expected = new Float32Array([1.0 + std * z0, -2.0 + std * z1, 0.5 + std * z2]);

  const out = addGaussianNoise(base, std);
  assertArrayClose(out[0], expected, 1e-12);

  (Math as any).random = orig; // restore
});

run();