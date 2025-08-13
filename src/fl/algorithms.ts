import {Weights, FLConfig} from "./types";

// Helpers
export function diffWeights(newW: Weights, oldW: Weights): Weights {
  return newW.map((nw, li) => {
    const ow = oldW[li];
    const d = new Float32Array(nw.length);
    for (let i=0;i<nw.length;i++) d[i] = nw[i] - ow[i];
    return d;
  });
}
export function addScaled(a: Weights, b: Weights, s: number): Weights {
  return a.map((aw, li) => {
    const bw = b[li];
    const out = new Float32Array(aw.length);
    for (let i=0;i<aw.length;i++) out[i] = aw[i] + s * bw[i];
    return out;
  });
}
export function zerosLike(w: Weights): Weights {
  return w.map(v => new Float32Array(v.length));
}

// Aggregation kernels
export function aggregateDeltas(
  deltas: {delta: Weights; weight: number}[],
  weighted: boolean
): Weights {
  const base = zerosLike(deltas[0].delta);
  let totalW = 0;
  for (const {delta, weight} of deltas) {
    const w = weighted ? weight : 1;
    totalW += w;
    for (let li=0;li<base.length;li++) {
      const b = base[li], d = delta[li];
      for (let i=0;i<b.length;i++) b[i] += w * d[i];
    }
  }
  // average
  for (let li=0;li<base.length;li++) {
    const b = base[li];
    for (let i=0;i<b.length;i++) b[i] /= totalW;
  }
  return base;
}

export function applyProximalGradient(
  // For FedProx, you account for μ(w - w0) inside local training. This helper is here
  // in case you choose to implement it as a server-side correction (less faithful).
  delta: Weights, w: Weights, w0: Weights, mu: number, steps: number
): Weights {
  if (!mu || mu <= 0) return delta;
  const prox = w.map((wi, li) => {
    const p = new Float32Array(wi.length);
    const di = w0[li];
    for (let i=0;i<wi.length;i++) p[i] = mu * (wi[i] - di[i]);
    return p;
  });
  // crude: subtract prox/steps from delta; prefer client-side implementation
  return addScaled(delta, prox, -1/Math.max(1, steps));
}
