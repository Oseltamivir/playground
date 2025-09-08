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
