import {ClientData} from "./types";

// Reproducible symmetric-Dirichlet sampler with a correct Gamma RNG.
// - Uses xorshift32 PRNG seeded at 3407 (persists across calls).
// - Marsaglia–Tsang for shape >= 1; boost trick for shape < 1.
// - No node modules.
export function sampleDirichlet(k: number, alpha: number): number[] {
  // persistent state on the function object for reproducibility across calls
  const self: any = sampleDirichlet;
  if (self._state === undefined) self._state = 3407 >>> 0; // fixed seed
  if (self._gaussSpare === undefined) self._gaussSpare = null;

  let state: number = self._state >>> 0;

  // xorshift32 -> U(0,1)
  function rand(): number {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    const u = (state >>> 0) / 4294967296; // [0,1)
    return u > 0 ? u : 1 / 4294967296;    // avoid exact 0 for logs
  }

  // Box–Muller -> N(0,1), with spare reuse
  function randn(): number {
    if (self._gaussSpare != null) {
      const z = self._gaussSpare;
      self._gaussSpare = null;
      return z;
    }
    const u1 = rand(), u2 = rand();
    const r = Math.sqrt(-2.0 * Math.log(u1));
    const theta = 2.0 * Math.PI * u2;
    const z0 = r * Math.cos(theta);
    const z1 = r * Math.sin(theta);
    self._gaussSpare = z1;
    return z0;
  }

  // Gamma(shape=a, scale=1)
  function gamma1(a: number): number {
    const EPS = 1e-12;
    if (!(a > 0)) a = EPS;
    if (a < 1) {
      // Boosting trick: Gamma(a) = Gamma(a+1) * U^(1/a)
      const u = rand();
      return gamma1(a + 1) * Math.pow(u, 1 / a);
    }
    // Marsaglia–Tsang (2000)
    const d = a - 1 / 3;
    const c = 1 / Math.sqrt(9 * d);
    while (true) {
      const x = randn();
      let v = 1 + c * x;
      if (v <= 0) continue;
      v = v * v * v;
      const u = rand();
      if (u < 1 - 0.0331 * (x * x) * (x * x)) return d * v;
      if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
    }
  }

  const a = alpha > 0 ? alpha : 1e-8;
  const xs: number[] = new Array(k);
  let sum = 0;
  for (let i = 0; i < k; i++) {
    const x = gamma1(a);
    xs[i] = x;
    sum += x;
  }
  if (!(sum > 0)) {
    // Fallback to uniform (extremely unlikely)
    for (let i = 0; i < k; i++) xs[i] = 1 / k;
  } else {
    for (let i = 0; i < k; i++) xs[i] /= sum;
  }

  self._state = state >>> 0; // persist PRNG state
  return xs;
}


export function makeClientsFromXY(
  X: number[][], y: number[], numClasses: number,
  numClients: number, batchSize: number, alpha: number
): ClientData[] {
  // Group indices by class
  var byClass: number[][] = [];
  for (var c = 0; c < numClasses; c++) byClass.push([]);
  for (var i = 0; i < y.length; i++) byClass[y[i]].push(i);

  // Shuffle within class
  for (var k = 0; k < numClasses; k++) {
    byClass[k].sort(function(){ return Math.random() - 0.5; });
  }

  var clients: ClientData[] = [];
  var classPtrs: number[] = [];
  for (var kk = 0; kk < numClasses; kk++) classPtrs.push(0);

  var total = y.length;
  var targetPerClient = Math.floor(total / Math.max(1, numClients));

  for (var cId = 0; cId < numClients; cId++) {
    var p = sampleDirichlet(numClasses, alpha);
    var takePerClass: number[] = [];
    for (var cc = 0; cc < numClasses; cc++) {
      var tpc = Math.round(p[cc] * targetPerClient);
      if (tpc < 0) tpc = 0;
      takePerClass.push(tpc);
    }

    var idxs: number[] = [];
    for (var cl = 0; cl < numClasses; cl++) {
      var start = classPtrs[cl];
      var end = Math.min(byClass[cl].length, start + takePerClass[cl]);
      for (var t = start; t < end; t++) idxs.push(byClass[cl][t]);
      classPtrs[cl] = end;
    }

    // Top-up if rounding left us short.
    var need = targetPerClient - idxs.length;
    var rot = 0;
    while (need > 0 && rot < 100000) {
      var which = rot % numClasses;
      if (classPtrs[which] < byClass[which].length) {
        idxs.push(byClass[which][classPtrs[which]]);
        classPtrs[which]++;
        need--;
      }
      rot++;
    }

    // Build batches
    var batches: {x:number[][]; y:number[]}[] = [];
    for (var b = 0; b < idxs.length; b += batchSize) {
      var sl = idxs.slice(b, b + batchSize);
      var bx: number[][] = [];
      var byArr: number[] = [];
      for (var s = 0; s < sl.length; s++) {
        var ii = sl[s];
        bx.push(X[ii]);
        byArr.push(y[ii]);
      }
      batches.push({x: bx, y: byArr});
    }

    // Class histogram for UI
    var hist: number[] = [];
    for (var hc = 0; hc < numClasses; hc++) hist.push(0);
    for (var h = 0; h < idxs.length; h++) hist[y[idxs[h]]]++;

    clients.push({id: cId, batches: batches, size: idxs.length, classHist: hist});
  }

  return clients;
}
