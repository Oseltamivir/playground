import { ServerAdam } from "../../src/fl/optimizers";
import { Weights } from "../../src/fl/types";
import { test, run, assertClose, assertArrayClose } from "../_util/harness";

// helpers
function W(arrs: number[][]): Weights {
  return arrs.map(a => new Float32Array(a));
}
function cloneW(w: Weights): Weights {
  return w.map(v => new Float32Array(v));
}

// Test 1: One-step Adam matches the paper’s closed form on constant grad g=1.
// m1=0.1, v1=0.001; mhat=1, vhat=1 ⇒ w1 = w0 - lr * 1/(sqrt(1)+eps) ≈ -lr
test("Adam: t=1 bias-corrected step equals -lr for constant grad=1", () => {
  const opt = new ServerAdam(0.01, 0.9, 0.999, 1e-8);
  const w0 = W([[0]]);
  const g  = W([[1]]);
  const w1 = opt.step(w0, g);
  assertClose(w1[0][0], -0.01, 1e-8);
});


// Test 2: Simple convex quadratic f(w)=0.5 * ||w||^2 with grad = w.
// Adam should drive weights toward 0 (not necessarily monotone every step).
test("Adam: converges toward 0 on quadratic (grad = w)", () => {
  const opt = new ServerAdam(0.02, 0.9, 0.999, 1e-8); // slightly larger lr for speed
  let w = W([[1, -2, 3]]);
  const initial = Array.from(w[0]);
  const initialNorm2 = initial.reduce((s, x) => s + x * x, 0);
  for (let t = 0; t < 400; t++) {
    const grad: Weights = cloneW(w); // grad = w
    w = opt.step(w, grad);
  }
  const finalArr = Array.from(w[0]);
  const finalNorm2 = finalArr.reduce((s, x) => s + x * x, 0);
  // in practice should shrink orders of magnitude
  if (!(finalNorm2 < initialNorm2 * 0.05)) {
    throw new Error("Did not shrink enough: final=" + finalNorm2 + " initial=" + initialNorm2);
  }
});


run();