import {Weights} from "./types";

// Simple server-side Adam on aggregated gradient (or weight delta)
export class ServerAdam {
  private m: Weights|null = null;
  private v: Weights|null = null;
  private t = 0;

  constructor(
    private lr = 0.01,
    private beta1 = 0.9,
    private beta2 = 0.999,
    private eps = 1e-8
  ) {}

  step(weights: Weights, grad: Weights): Weights {
    this.t += 1;
    if (!this.m) this.m = grad.map(g => new Float32Array(g.length));
    if (!this.v) this.v = grad.map(g => new Float32Array(g.length));

    const out: Weights = weights.map((w, li) => {
      const g = grad[li];
      const m = this!.m![li], v = this!.v![li];
      const o = new Float32Array(w.length);

      for (let i=0;i<w.length;i++) {
        m[i] = this.beta1*m[i] + (1-this.beta1)*g[i];
        v[i] = this.beta2*v[i] + (1-this.beta2)*g[i]*g[i];
        const mhat = m[i] / (1 - Math.pow(this.beta1, this.t));
        const vhat = v[i] / (1 - Math.pow(this.beta2, this.t));
        o[i] = w[i] - this.lr * (mhat / (Math.sqrt(vhat) + this.eps));
      }
      return o;
    });

    return out;
  }
}
