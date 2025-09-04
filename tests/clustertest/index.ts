import { dot, norm, cosineSim, l2Dist2, kMeans } from "../../src/fl/cluster";
import { test, run, assertClose, assertArrayClose } from "../_util/harness";

function F(a: number[]): Float32Array { return new Float32Array(a); }

// Test 1: Self-explanatory tests for cluster.ts functions
test("dot / norm / cosineSim / l2Dist2", () => {
  const a = F([1, 2, 3]);
  const b = F([4, 5, 6]);
  // dot = 1*4 + 2*5 + 3*6 = 32
  assertClose(dot(a, b), 32);
  // ||a|| = sqrt(14)
  assertClose(norm(a), Math.sqrt(14));
  // cosine(a,a) = 1; cosine(e1,e2)=0
  assertClose(cosineSim(a, a), 1);
  assertClose(cosineSim(F([1,0]), F([0,1])), 0);
  // l2Dist2([1,2],[3,5]) = (2^2 + 3^2) = 13
  assertClose(l2Dist2(F([1,2]), F([3,5])), 13);
});

// Test 2: K-Means on simple datasets
test("kMeans (L2): two obvious clusters", () => {
  const pts = [F([0,0]), F([0,1]), F([5,5]), F([5,6])];
  const { assignments, centroids } = kMeans(pts, 2, "l2", 50, 42);
  // Expect two groups of size 2 and centroids near [0,0.5] and [5,5.5]
  const c0 = centroids[0], c1 = centroids[1];
  const aSum = assignments.reduce((s, x) => s + x, 0); // should be 2 if two in cluster 1
  if (!(aSum === 2)) throw new Error("Unexpected assignment pattern: " + assignments.join(","));
  const expectA = [0, 0.5], expectB = [5, 5.5];
  const closeToA = (v: Float32Array) => Math.hypot(v[0]-expectA[0], v[1]-expectA[1]) < 1e-3;
  const closeToB = (v: Float32Array) => Math.hypot(v[0]-expectB[0], v[1]-expectB[1]) < 1e-3;
  if (!((closeToA(c0) && closeToB(c1)) || (closeToB(c0) && closeToA(c1)))) {
    throw new Error("Centroids not at expected means.");
  }
});

// Test 3: K-Means on simple datasets with cosine distance
test("kMeans (cosine): separates directions", () => {
  const pts = [F([1,0]), F([0.99,0.01]), F([0,1]), F([0.01,0.99])];
  const { assignments, centroids } = kMeans(pts, 2, "cosine", 50, 7);
  // two clusters: near x-axis and y-axis directions
  function dir(v: Float32Array) { return v[0] >= v[1] ? "x" : "y"; }
  const dirs = centroids.map(dir).sort().join("");
  if (!(dirs === "xy")) throw new Error("Cosine centroids not in expected directions: " + dirs);
});

run();