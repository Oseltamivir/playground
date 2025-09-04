import {Weights} from "./types";

// L2 clip each tensor group to given norm, return clipped copy and (optional) noise
export function clipUpdate(delta: Weights, clip: number): Weights {
  if (!isFinite(clip) || clip <= 0) return delta;
  // Compute global L2 of concatenated vectors
  let sq = 0;
  delta.forEach(w => { for (let i=0;i<w.length;i++) sq += w[i]*w[i]; });
  const norm = Math.sqrt(sq) || 1e-12;
  const scale = Math.min(1, clip / norm);
  if (scale === 1) return delta;
  return delta.map(w => {
    const out = new Float32Array(w.length);
    for (let i=0;i<w.length;i++) out[i] = w[i] * scale;
    return out;
  });
}

// Add Gaussian noise to each tensor group, return noisy copy
export function addGaussianNoise(delta: Weights, std: number): Weights {
  if (!isFinite(std) || std <= 0) return delta;
  return delta.map(w => {
    const out = new Float32Array(w.length);
    for (let i=0;i<w.length;i++) {
      // Box-Muller
      const u = Math.random(), v = Math.random();
      const n = Math.sqrt(-2*Math.log(u)) * Math.cos(2*Math.PI*v);
      out[i] = w[i] + std * n;
    }
    return out;
  });
}
