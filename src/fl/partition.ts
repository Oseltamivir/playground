import {ClientData} from "./types";

// Crude Dirichlet sampler using Gamma(alpha,1) approximations; fine for UI.
function sampleDirichlet(k: number, alpha: number): number[] {
  var out: number[] = [];
  var sum = 0;
  var a = (alpha > 0 ? alpha : 1e-3);
  for (var i = 0; i < k; i++) {
    // Box–Muller normal, transform to rough Gamma; good enough for visualization.
    var u = Math.random(), v = Math.random();
    var n = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    var x = (a - 1/3) * Math.pow(1 + n / Math.sqrt(9 * a - 3), 3);
    if (!isFinite(x) || x <= 0) x = 1e-6;
    out.push(x);
    sum += x;
  }
  // normalize
  for (var j = 0; j < k; j++) out[j] = out[j] / sum;
  return out;
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
