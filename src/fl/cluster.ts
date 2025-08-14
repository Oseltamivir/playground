export type Metric = "cosine" | "l2";

export function dot(a: Float32Array, b: Float32Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}
export function norm(a: Float32Array): number {
  return Math.sqrt(dot(a, a));
}
export function cosineSim(a: Float32Array, b: Float32Array): number {
  const na = norm(a) || 1e-12;
  const nb = norm(b) || 1e-12;
  return dot(a, b) / (na * nb);
}
export function l2Dist2(a: Float32Array, b: Float32Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    s += d * d;
  }
  return s;
}

function normalizeInPlace(v: Float32Array): void {
  const n = norm(v);
  const d = n || 1e-12;
  for (let i = 0; i < v.length; i++) v[i] /= d;
}

function clone(v: Float32Array): Float32Array {
  const o = new Float32Array(v.length);
  o.set(v);
  return o;
}

export function kMeans(
  vectors: Float32Array[],
  K: number,
  metric: Metric = "cosine",
  maxIters = 20,
  seed = 0
): { assignments: number[]; centroids: Float32Array[] } {
  const N = vectors.length;
  if (N === 0 || K <= 1) {
    // Manual fill for assignments
    const assignments: number[] = [];
    for (let i = 0; i < N; i++) assignments[i] = 0;
    return {
      assignments,
      centroids: [average(vectors.length ? vectors : [new Float32Array(0)])],
    };
  }
  K = Math.min(K, N);

  // Preprocess for cosine metric (work on normalized copies for stability).
  const data = metric === "cosine" ? vectors.map(clone) : vectors.map(clone);
  if (metric === "cosine") for (const v of data) normalizeInPlace(v);

  // Init centroids with simple seeded shuffle pick
  // Init centroids with simple seeded shuffle pick
  const idxs: number[] = [];
  for (let i = 0; i < N; i++) idxs[i] = i;
  seededShuffle(idxs, seed);
  let centroids = [];
  for (let i = 0; i < K; i++) centroids[i] = clone(data[idxs[i]]);

  // Manual fill for assignments
  let assignments: number[] = [];
  for (let i = 0; i < N; i++) assignments[i] = 0;

  for (let it = 0; it < maxIters; it++) {
    let changed = false;

    // Assign step
    for (let i = 0; i < N; i++) {
      let best = 0;
      let bestScore = Infinity;
      for (let c = 0; c < K; c++) {
        let score: number;
        if (metric === "cosine") {
          // distance = 1 - cosine
          score = 1 - Math.max(-1, Math.min(1, dot(data[i], centroids[c])));
        } else {
          score = l2Dist2(data[i], centroids[c]);
        }
        if (score < bestScore) {
          bestScore = score;
          best = c;
        }
      }
      if (assignments[i] !== best) {
        assignments[i] = best;
        changed = true;
      }
    }

    // Update step
    const sums: Float32Array[] = [];
    const counts: number[] = [];
    for (let c = 0; c < K; c++) {
      sums[c] = new Float32Array(data[0].length);
      counts[c] = 0;
    }
    for (let i = 0; i < N; i++) {
      const a = assignments[i];
      const v = data[i];
      for (let j = 0; j < v.length; j++) sums[a][j] += v[j];
      counts[a]++;
    }
    for (let c = 0; c < K; c++) {
      if (counts[c] > 0) {
        for (let j = 0; j < sums[c].length; j++) sums[c][j] /= counts[c];
        if (metric === "cosine") normalizeInPlace(sums[c]);
        centroids[c] = sums[c];
      } // else keep previous centroid
    }

    if (!changed) break;
  }
  return { assignments, centroids };
} // <-- Add this missing closing brace

function average(vs: Float32Array[]): Float32Array {
  let d = 0;
  if (vs.length > 0 && vs[0]) d = vs[0].length;
  const out = new Float32Array(d);
  if (vs.length === 0) return out;
  for (let i = 0; i < vs.length; i++) {
    const v = vs[i];
    for (let j = 0; j < d; j++) {
      out[j] += v[j];
    }
  }
  for (let j = 0; j < d; j++) {
    out[j] /= vs.length;
  }
  return out;
}

function seededShuffle<T>(arr: T[], seed: number) {
  let s = seed || 0x9e3779b1;
  const rand = () => {
    s ^= s << 13; s ^= s >>> 17; s ^= s << 5;
    return (s >>> 0) / 0xffffffff;
  };
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
}