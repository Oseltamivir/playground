export type Metric = "cosine" | "l2";

// Compute dot product between two equal-length Float32Arrays.
export function dot(a: Float32Array, b: Float32Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

// Compute the Euclidean norm (L2) of a vector.
export function norm(a: Float32Array): number {
  return Math.sqrt(dot(a, a));
}

// Cosine similarity between two vectors (returns in [-1, 1]).
export function cosineSim(a: Float32Array, b: Float32Array): number {
  // Tiny epsilon to avoid division by zero.
  const na = norm(a) || 1e-12;
  const nb = norm(b) || 1e-12;
  return dot(a, b) / (na * nb);
}

// Squared L2 distance between two vectors
export function l2Dist2(a: Float32Array, b: Float32Array): number {
  // Accumulator for squared differences.
  let s = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    s += d * d;
  }
  return s;
}

// In-place L2 normalization of a vector
function normalizeInPlace(v: Float32Array): void {
  const n = norm(v);
  const d = Math.max(n, 1e-12);
  // Scale all components by 1 / ||v||.
  for (let i = 0; i < v.length; i++) v[i] /= d;
}

// Deepcopy
function clone(v: Float32Array): Float32Array {
  const o = new Float32Array(v.length);
  o.set(v);
  return o;
}

// K-Means clustering over Float32Array vectors.
// Returns the cluster assignment per vector and the learned centroids.
// - vectors: data points (each Float32Array is a D-dimensional point)
// - K: number of clusters (will be clipped to N when K > N)
// - metric: "cosine" (uses 1 - cosine similarity as distance) or "l2" (squared L2)
// - maxIters: hard cap on Lloyd iterations
// - seed: used for deterministic centroid initialization via seededShuffle
export function kMeans(
  vectors: Float32Array[],
  K: number,
  metric: Metric = "cosine",
  maxIters = 20,
  seed = 0
): { assignments: number[]; centroids: Float32Array[] } {
  const N = vectors.length;

  // For cosine distance, we work on normalized copies for numerical stability.
  // Cloning prevents mutating the input "vectors".
  const data = metric === "cosine" ? vectors.map(clone) : vectors.map(clone);
  if (metric === "cosine") for (const v of data) normalizeInPlace(v);

  // --- Initialization (choose K distinct points via seeded shuffle) ---
  // Build index list 0..N-1.
  const idxs: number[] = [];
  for (let i = 0; i < N; i++) idxs[i] = i;
  
  seededShuffle(idxs, seed);

  // Initialize centroids by copying the first K shuffled points.
  let centroids = [];
  for (let i = 0; i < K; i++) {
    centroids[i] = clone(data[idxs[i]]);
  }

  // Initialize assignments array with zeros (cluster 0 by default).
  let assignments: number[] = [];
  for (let i = 0; i < N; i++) {
    assignments[i] = 0;
  }
  // Update until convergence or maxIters
  for (let it = 0; it < maxIters; it++) {
    // Track whether any point changed its assigned cluster.
    let changed = false;

    // Assign step: assign each point to nearest centroid
    for (let i = 0; i < N; i++) {
      // Best cluster index and best (min) distance so far.
      let best = 0;
      let bestScore = Infinity;

      // Compare to each centroid.
      for (let c = 0; c < K; c++) {
        let score: number;
        if (metric === "cosine") {
          // With normalized vectors, cosine distance = 1 - cosine similarity.
          // Clamp the dot product to [-1, 1] to ensure stability
          score = 1 - Math.max(-1, Math.min(1, dot(data[i], centroids[c])));
        } else {
          // Use squared Euclidean distance for L2.
          score = l2Dist2(data[i], centroids[c]);
        }
        // Keep the closest centroid (smallest distance).
        if (score < bestScore) {
          bestScore = score;
          best = c;
        }
      }

      // Update assignment and mark if anything changed.
      if (assignments[i] !== best) {
        assignments[i] = best;
        changed = true;
      }
    }

    // ----- Update step: recompute each centroid as mean of its assigned points -----
    // Accumulators (sum vectors) per cluster and their counts.
    const sums: Float32Array[] = [];
    const counts: number[] = [];
    for (let c = 0; c < K; c++) {
      // Allocate zero vector for sums (dimension = data[0].length).
      sums[c] = new Float32Array(data[0].length);
      counts[c] = 0;
    }

    // Accumulate sums and counts per assigned cluster.
    for (let i = 0; i < N; i++) {
      const a = assignments[i];
      const v = data[i];
      for (let j = 0; j < v.length; j++) sums[a][j] += v[j];
      counts[a]++;
    }

    // Finalize new centroids: average (and renormalize for cosine).
    for (let c = 0; c < K; c++) {
      if (counts[c] > 0) {
        // Divide by count to get mean.
        for (let j = 0; j < sums[c].length; j++) {
          sums[c][j] /= counts[c];
        }
        if (metric === "cosine") normalizeInPlace(sums[c]);
        // Replace the centroid.
        centroids[c] = sums[c];
      } // If a cluster has no points, keep the previous centroid as-is.
    }

    // Early exit if assignments stabilized.
    if (!changed) break;
  }

  // Return final assignments and centroids.
  return { assignments, centroids };
}

// Deterministic in-place array shuffle using a simple xorshift32 PRNG.
// Produces a Fisher–Yates shuffle controlled by "seed".
function seededShuffle<T>(arr: T[], seed: number) {
  // Initialize state; default constant if seed=0 for reproducibility.
  let s = seed || 0x9e3779b1;
  // xorshift32-based PRNG; returns float in [0, 1).
  const rand = () => {
    s ^= s << 13; s ^= s >>> 17; s ^= s << 5;
    return (s >>> 0) / 0xffffffff;
  };
  // Fisher–Yates from end to start.
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
}