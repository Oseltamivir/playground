import {ClientData, FLConfig, FLCallbacks, Weights} from "./types";
import {aggregateDeltas, diffWeights, addScaled} from "./algorithms";
import {clipUpdate, addGaussianNoise} from "./dp";
import {ServerAdam} from "./optimizers";

export function simulateFederatedTraining(
  clients: ClientData[],
  cfg: FLConfig,
  cb: FLCallbacks,
  numRounds: number,
  onDone?: () => void
): void {
  // Clone starting global weights
  var w = cb.cloneWeights();

  // Avoid nullish coalescing
  var serverLR = (cfg.serverLR !== undefined && cfg.serverLR !== null) ? cfg.serverLR : 0.01;
  var beta1    = (cfg.beta1    !== undefined && cfg.beta1    !== null) ? cfg.beta1    : 0.9;
  var beta2    = (cfg.beta2    !== undefined && cfg.beta2    !== null) ? cfg.beta2    : 0.999;
  var eps      = (cfg.eps      !== undefined && cfg.eps      !== null) ? cfg.eps      : 1e-8;
  var adam = (cfg.algo === "FedAdam") ? new ServerAdam(serverLR, beta1, beta2, eps) : null;

  // Optional: show client class histograms once
  if (cb.onClientHistograms) {
    var hists: number[][] = [];
    for (var ci = 0; ci < clients.length; ci++) {
      hists.push(clients[ci].classHist ? clients[ci].classHist! : []);
    }
    cb.onClientHistograms(hists);
  }

  var r = 1;

  function doRound() {
    // Sample clients
    var k = Math.max(1, Math.round(cfg.clientFrac * clients.length));
    var shuffled = clients.slice(0);
    shuffled.sort(function(){ return Math.random() - 0.5; });

    var dropout = (cfg.clientDropout !== undefined && cfg.clientDropout !== null) ? cfg.clientDropout : 0;
    var roundClients: ClientData[] = [];
    for (var i = 0; i < k && i < shuffled.length; i++) {
      if (Math.random() > dropout) roundClients.push(shuffled[i]);
    }

    // Local training
    var deltas: {delta: Weights; weight: number}[] = [];
    for (var rc = 0; rc < roundClients.length; rc++) {
      var c = roundClients[rc];

      // copy w to wLocal
      var wLocal: Weights = [];
      for (var li = 0; li < w.length; li++) {
        var copy = new Float32Array(w[li].length);
        for (var j = 0; j < w[li].length; j++) copy[j] = w[li][j];
        wLocal.push(copy);
      }

      for (var e = 0; e < cfg.localEpochs; e++) {
        wLocal = cb.localTrainOneEpoch(w, c, cfg); // FedProx uses w as w0
      }

      var delta = diffWeights(wLocal, w);

      // Client-level DP
      if (cfg.dpClientLevel && cfg.dpClipNorm && cfg.dpClipNorm > 0) {
        delta = clipUpdate(delta, cfg.dpClipNorm);
        var dpNoiseMult = (cfg.dpNoiseMult !== undefined && cfg.dpNoiseMult !== null) ? cfg.dpNoiseMult : 0;
        var sigma = dpNoiseMult * cfg.dpClipNorm;
        if (sigma > 0) delta = addGaussianNoise(delta, sigma);
      }

      deltas.push({delta: delta, weight: c.size});
    }

    // Aggregate updates
    var agg = aggregateDeltas(deltas, !!cfg.weightedAggregation);

    // Server-level DP
    if (!cfg.dpClientLevel && cfg.dpClipNorm && cfg.dpClipNorm > 0) {
      agg = clipUpdate(agg, cfg.dpClipNorm);
      var dpNoiseMult2 = (cfg.dpNoiseMult !== undefined && cfg.dpNoiseMult !== null) ? cfg.dpNoiseMult : 0;
      var sigma2 = dpNoiseMult2 * cfg.dpClipNorm;
      if (sigma2 > 0) agg = addGaussianNoise(agg, sigma2);
    }

    // Server update
    if (cfg.algo === "FedAdam" && adam) {
      // interpret agg as gradient; deep copy
      var grad: Weights = [];
      for (var li2 = 0; li2 < agg.length; li2++) {
        var a = agg[li2];
        var gcopy = new Float32Array(a.length);
        for (var jj = 0; jj < a.length; jj++) gcopy[jj] = -a[jj];
        grad.push(gcopy);
      }
      w = adam.step(w, grad);
    } else {
      // FedAvg / FedProx: w <- w + mean(delta)
      w = addScaled(w, agg, 1.0);
    }

    cb.setWeights(w);

    // Eval & UI
    var evalRes = cb.evalGlobal();
    if (cb.onRoundEnd) {
      var comm = estimateCommBytes(deltas, roundClients.length);
      cb.onRoundEnd({
        round: r,
        participating: roundClients.length,
        globalAcc: evalRes.acc,
        globalLoss: evalRes.loss,
        commBytes: comm
      }, w);
    }

    r++;
    if (r <= numRounds) {
      // keep UI responsive without Promises
      requestAnimationFrame(doRound);
    } else {
      if (onDone) onDone();
    }
  }

  // kick off
  requestAnimationFrame(doRound);
}

function estimateCommBytes(deltas: {delta: Weights}[], k: number): number {
  if (!deltas || deltas.length === 0) return 0;
  var nFloats = 0;
  var first = deltas[0].delta;
  for (var li = 0; li < first.length; li++) nFloats += first[li].length;
  return (nFloats * 4) * (k + 1); // uploads + one download
}
