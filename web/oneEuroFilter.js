// One Euro Filter (Casiez, Roussel, Vogel — CHI 2012).
// https://gery.casiez.net/1euro/
//
// Low-pass filter for noisy real-time signals. The cutoff frequency rises with
// the signal's velocity, so the filter smooths jitter at rest while staying
// responsive during fast motion.
//
// Tunables:
//   mincutoff (Hz) - cutoff at zero velocity. Lower = smoother (more lag).
//   beta           - how aggressively cutoff increases with |velocity|.
//                    Higher = more responsive to fast motion.
//   dcutoff (Hz)   - cutoff on the derivative estimate itself (usually 1.0).

export class OneEuroFilter {
  constructor({ mincutoff = 1.0, beta = 0.0, dcutoff = 1.0 } = {}) {
    this.mincutoff = mincutoff;
    this.beta = beta;
    this.dcutoff = dcutoff;
    this.lastValue = null;
    this.lastDerivative = 0;
    this.lastTime = null;
  }

  reset() {
    this.lastValue = null;
    this.lastDerivative = 0;
    this.lastTime = null;
  }

  setParams({ mincutoff, beta }) {
    if (mincutoff !== undefined) this.mincutoff = mincutoff;
    if (beta !== undefined) this.beta = beta;
  }

  _alpha(cutoff, dtSec) {
    const tau = 1.0 / (2 * Math.PI * cutoff);
    return 1.0 / (1.0 + tau / dtSec);
  }

  filter(value, tMs) {
    if (this.lastTime === null) {
      this.lastValue = value;
      this.lastTime = tMs;
      return value;
    }
    const dt = (tMs - this.lastTime) / 1000;
    if (dt <= 0) return this.lastValue;

    const derivative = (value - this.lastValue) / dt;
    const aD = this._alpha(this.dcutoff, dt);
    const smoothedDerivative = aD * derivative + (1 - aD) * this.lastDerivative;

    const cutoff = this.mincutoff + this.beta * Math.abs(smoothedDerivative);
    const a = this._alpha(cutoff, dt);
    const smoothed = a * value + (1 - a) * this.lastValue;

    this.lastValue = smoothed;
    this.lastDerivative = smoothedDerivative;
    this.lastTime = tMs;
    return smoothed;
  }
}

/**
 * Vectorized One Euro Filter — same math, applied independently per component
 * across a typed-array signal. Avoids the per-element object overhead of N
 * scalar filters and keeps state in flat Float32Arrays.
 */
export class VectorOneEuroFilter {
  constructor(n, { mincutoff = 1.0, beta = 0.0, dcutoff = 1.0 } = {}) {
    this.n = n;
    this.mincutoff = mincutoff;
    this.beta = beta;
    this.dcutoff = dcutoff;
    this.values = new Float32Array(n);
    this.derivatives = new Float32Array(n);
    this.lastTime = null;
  }

  reset() {
    this.values.fill(0);
    this.derivatives.fill(0);
    this.lastTime = null;
  }

  setParams({ mincutoff, beta }) {
    if (mincutoff !== undefined) this.mincutoff = mincutoff;
    if (beta !== undefined) this.beta = beta;
  }

  /**
   * @param {Float32Array} input - length n
   * @param {number} tMs
   * @param {Float32Array} [out] - length n; written in place. If omitted, returns this.values.
   */
  filter(input, tMs, out) {
    const n = this.n;
    if (this.lastTime === null) {
      this.values.set(input);
      this.lastTime = tMs;
      if (out) out.set(input);
      return out || this.values;
    }
    const dt = (tMs - this.lastTime) / 1000;
    if (dt <= 0) {
      if (out) out.set(this.values);
      return out || this.values;
    }
    const tauD = 1 / (2 * Math.PI * this.dcutoff);
    const aD = 1 / (1 + tauD / dt);
    const mincutoff = this.mincutoff;
    const beta = this.beta;
    const invDt = 1 / dt;
    const twoPi = 2 * Math.PI;
    for (let i = 0; i < n; i++) {
      const xi = input[i];
      const deriv = (xi - this.values[i]) * invDt;
      const smDeriv = aD * deriv + (1 - aD) * this.derivatives[i];
      this.derivatives[i] = smDeriv;
      const cutoff = mincutoff + beta * Math.abs(smDeriv);
      const tau = 1 / (twoPi * cutoff);
      const a = 1 / (1 + tau / dt);
      const sm = a * xi + (1 - a) * this.values[i];
      this.values[i] = sm;
      if (out) out[i] = sm;
    }
    this.lastTime = tMs;
    return out || this.values;
  }
}
