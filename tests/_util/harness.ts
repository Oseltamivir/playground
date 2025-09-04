/* Minimal zero-deps test harness for Node + TS 2.9 */
declare var process: any;

export type TestFn = () => void;

let failures = 0;
const tests: { name: string; fn: TestFn }[] = [];

export function test(name: string, fn: TestFn): void {
  tests.push({ name, fn });
}

function approx(a: number, b: number, eps: number = 1e-6): boolean {
  const scale = 1 + Math.max(Math.abs(a), Math.abs(b));
  return Math.abs(a - b) <= eps * scale;
}

export function assert(cond: boolean, msg: string): void {
  if (!cond) throw new Error(msg);
}

export function assertClose(a: number, b: number, eps: number = 1e-6): void {
  if (!approx(a, b, eps)) {
    throw new Error("Expected " + a + " ≈ " + b + " (eps=" + eps + ")");
  }
}

export function assertArrayClose(a: ArrayLike<number>, b: ArrayLike<number>, eps: number = 1e-6): void {
  if (a.length !== b.length) throw new Error("Length mismatch: " + a.length + " vs " + b.length);
  for (let i = 0; i < a.length; i++) {
    if (!approx(a[i], b[i], eps)) throw new Error("Index " + i + " mismatch: " + a[i] + " vs " + b[i]);
  }
}

export function run(): void {
  for (let i = 0; i < tests.length; i++) {
    const t = tests[i];
    try {
      t.fn();
      console.log("✓ " + t.name);
    } catch (e) {
      failures++;
      console.error("✗ " + t.name + " -> " + (e && (e as any).message ? (e as any).message : e));
    }
  }
  if (failures > 0) {
    console.error("FAILED: " + failures + " test(s).");
    if (typeof process !== "undefined" && process && process.exit) process.exit(1);
  } else {
    console.log("All tests passed.");
  }
}