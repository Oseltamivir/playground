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

  const a = alpha > 0 ? alpha : 1e-3;
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
  X: number[][],
  y: number[],
  numClasses: number,
  numClients: number,
  batchSize: number,
  alpha: number,
  balance: number = 1.0,
  seed?: number
): ClientData[] {
  // ---- Seeded RNG (xorshift32) with Math.random fallback ----
  const useSeed = seed !== undefined && seed !== null;
  let rngState = (seed as number) >>> 0;
  let gaussSpare: number | null = null;

  function rand(): number {
    if (!useSeed) return Math.random();
    rngState ^= rngState << 13;
    rngState ^= rngState >>> 17;
    rngState ^= rngState << 5;
    const u = (rngState >>> 0) / 4294967296;
    return u > 0 ? u : 1 / 4294967296;
  }
  function randint(n: number): number { return Math.floor(rand() * n); }
  function shuffle<T>(arr: T[]): void {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = randint(i + 1);
      const t = arr[i]; arr[i] = arr[j]; arr[j] = t;
    }
  }
  function randn(): number {
    if (!useSeed) {
      // Box–Muller via Math.random
      const u1 = Math.random() || 1 / 2 ** 32, u2 = Math.random();
      return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }
    if (gaussSpare != null) { const z = gaussSpare; gaussSpare = null; return z; }
    const u1 = rand(), u2 = rand();
    const r = Math.sqrt(-2 * Math.log(u1)), th = 2 * Math.PI * u2;
    const z0 = r * Math.cos(th), z1 = r * Math.sin(th);
    gaussSpare = z1; return z0;
  }

  // Reset the global Dirichlet RNG if caller wants determinism across rebuilds
  if (useSeed) {
    (sampleDirichlet as any)._state = (seed! ^ 0x9e3779b9) >>> 0;
    (sampleDirichlet as any)._gaussSpare = null;
  }

  // ---- Helpers: Gamma & Dirichlet (local, seeded) for client-size weights ----
  function gamma1(a: number): number {
    const EPS = 1e-12;
    a = a > 0 ? a : EPS;
    if (a < 1) {
      const u = rand();
      return gamma1(a + 1) * Math.pow(u, 1 / a); // boost trick
    }
    const d = a - 1 / 3, c = 1 / Math.sqrt(9 * d);
    while (true) {
      const x = randn();
      let v = 1 + c * x; if (v <= 0) continue;
      v = v * v * v;
      const u = rand();
      if (u < 1 - 0.0331 * (x * x) * (x * x)) return d * v;
      if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
    }
  }
  function dirichletSymWeights(k: number, a: number): number[] {
    const out = new Array<number>(k);
    let s = 0;
    for (let i = 0; i < k; i++) { out[i] = gamma1(a); s += out[i]; }
    if (s <= 0) return out.fill(1 / k);
    for (let i = 0; i < k; i++) out[i] /= s;
    return out;
  }

  // ---- Group indices by class & shuffle within each class (seeded) ----
  const byClass: number[][] = Array.from({ length: numClasses }, () => []);
  for (let i = 0; i < y.length; i++) byClass[y[i]].push(i);
  for (let c = 0; c < numClasses; c++) shuffle(byClass[c]);

  const total = y.length;
  const clients: ClientData[] = [];
  const classPtrs = new Array<number>(numClasses).fill(0);

  // ---- (1) Reserve: ensure each client gets >=1 sample when possible ----
  const initialIdxs: number[][] = Array.from({ length: numClients }, () => []);
  if (total >= numClients) {
    // round-robin across classes, starting at a random rotation for fairness
    let rot = useSeed ? randint(numClasses) : 0;
    for (let cId = 0; cId < numClients; cId++) {
      let tries = 0;
      while (tries < numClasses &&
             classPtrs[rot % numClasses] >= byClass[rot % numClasses].length) {
        rot++; tries++;
      }
      const cl = rot % numClasses;
      if (classPtrs[cl] < byClass[cl].length) {
        initialIdxs[cId].push(byClass[cl][classPtrs[cl]++]);
      }
      rot++;
    }
  } else {
    // Not enough samples for everyone: assign as many distinct as we can.
    let left = total, rot = useSeed ? randint(numClasses) : 0;
    for (let cId = 0; cId < numClients && left > 0; cId++) {
      for (let t = 0; t < numClasses; t++) {
        const cl = (rot + t) % numClasses;
        if (classPtrs[cl] < byClass[cl].length) {
          initialIdxs[cId].push(byClass[cl][classPtrs[cl]++]);
          left--; break;
        }
      }
      rot++;
    }
  }

  // ---- (2) Decide client sizes from a Dirichlet over clients (≠ class Dirichlet) ----
  const reservedPerClient = initialIdxs.map(v => v.length);
  const initialAssigned = reservedPerClient.reduce((a, b) => a + b, 0);
  const remainingTotal = Math.max(0, total - initialAssigned);

  const clientSizes = new Array<number>(numClients);
  if (remainingTotal === 0) {
    for (let i = 0; i < numClients; i++) clientSizes[i] = reservedPerClient[i];
  } else if (balance >= 0.999) {
    // Equal sizes + reserved
    const base = Math.floor(remainingTotal / numClients);
    let rem = remainingTotal - base * numClients;
    // randomize who gets the remainders for fairness
    const order = Array.from({ length: numClients }, (_, i) => i);
    shuffle(order);
    for (let i = 0; i < numClients; i++) {
      const add = i < rem ? 1 : 0;
      clientSizes[order[i]] = reservedPerClient[order[i]] + base + add;
    }
  } else {
    // Dirichlet weights over clients controlled by balance
    // map balance∈[0,1] → a_size: small => spiky, large => near-uniform
    const A_LO = 0.3, A_HI = 50.0;
    const a_size = A_LO + (A_HI - A_LO) * Math.max(0, Math.min(1, balance));
    const w = dirichletSymWeights(numClients, a_size);

    // Soft cap to avoid a single-client hog (still allows skew)
    const mu = remainingTotal / numClients;
    let capAdd = Math.max(1, Math.ceil(mu * (1 + 3 * (1 - balance)))); // up to 4×mean when balance=0
    if (capAdd * numClients < remainingTotal) capAdd = Math.ceil(remainingTotal / numClients);

    // Largest-remainder (Hamilton) apportionment with caps over the "additional" part
    const quotas = w.map(p => p * remainingTotal);
    const add = quotas.map(q => Math.min(Math.floor(q), capAdd));
    let used = add.reduce((a, b) => a + b, 0);
    let rem = remainingTotal - used;

    // Pre-shuffle for deterministic tie-breaking, then give out by fractional part
    const idx = Array.from({ length: numClients }, (_, i) => i);
    shuffle(idx);

    // ---- Helper for giving out remainders (seeded) ----
    const giveRemainders = (currCap: number, quotas: number[], add: number[], idx: number[]): void => {
      // sort by fractional remainder desc
      idx.sort((i, j) => (quotas[j] - Math.floor(quotas[j])) - (quotas[i] - Math.floor(quotas[i])));
      for (let t = 0; t < idx.length && rem > 0; t++) {
        const i = idx[t];
        if (add[i] < currCap) { add[i]++; rem--; }
      }
    };
    while (rem > 0) {
      const before = rem;
      giveRemainders(capAdd, quotas, add, idx);
      if (rem === before) { capAdd++; } // all at cap → relax and continue
    }

    for (let i = 0; i < numClients; i++) clientSizes[i] = reservedPerClient[i] + add[i];
  }

  // ---- Helper for multinomial pick with availability (seeded) ----
  const pickClass = (prob: number[], available: boolean[]): number => {
    let totalProb = 0;
    for (let c = 0; c < numClasses; c++) if (available[c]) totalProb += prob[c];
    if (totalProb <= 0) {
      // uniform over available
      const availIdx: number[] = [];
      for (let c = 0; c < numClasses; c++) if (available[c]) availIdx.push(c);
      if (availIdx.length === 0) return -1;
      return availIdx[randint(availIdx.length)];
    }
    const r = rand() * totalProb;
    let acc = 0;
    for (let c = 0; c < numClasses; c++) {
      if (!available[c]) continue;
      acc += prob[c];
      if (r <= acc) return c;
    }
    // fallback
    for (let c = numClasses - 1; c >= 0; c--) if (available[c]) return c;
    return -1;
  };

  // ---- (3) Build clients: class mixture comes solely from alpha (IID knob) ----
  for (let cId = 0; cId < numClients; cId++) {
    // draw per-client class proportions from your existing sampler (seeded above)
    const p = sampleDirichlet(numClasses, alpha);
    const target = clientSizes[cId];

    // Start with reserved one(s)
    const idxs: number[] = initialIdxs[cId].slice();

    // Fill remaining according to p, respecting per-class availability
    let need = Math.max(0, target - idxs.length);
    while (need > 0) {
      const available: boolean[] = new Array(numClasses);
      let anyAvail = false;
      for (let c = 0; c < numClasses; c++) {
        available[c] = classPtrs[c] < byClass[c].length;
        if (available[c]) anyAvail = true;
      }
      if (!anyAvail) break;
      const cl = pickClass(p, available);
      if (cl < 0) break;
      idxs.push(byClass[cl][classPtrs[cl]++]);
      need--;
    }

    // Build batches
    // Example: if input = [7, 3, 10, 5, 2], BS = 2,
    // Output = [7,3], [10,5], [2]
    const batches: { x: number[][]; y: number[] }[] = [];
    for (let b = 0; b < idxs.length; b += batchSize) {
      const sl = idxs.slice(b, b + batchSize);
      const bx: number[][] = [];
      const byArr: number[] = [];
      for (let s = 0; s < sl.length; s++) {
        const ii = sl[s];
        bx.push(X[ii]);
        byArr.push(y[ii]);
      }
      batches.push({ x: bx, y: byArr });
    }

    // Class histogram for UI
    const hist = new Array<number>(numClasses).fill(0);
    for (let h = 0; h < idxs.length; h++) hist[y[idxs[h]]]++;

    clients.push({ id: cId, batches, size: idxs.length, classHist: hist });
  }

  return clients;
}
