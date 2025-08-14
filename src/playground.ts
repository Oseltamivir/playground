/* Copyright 2016 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

import * as nn from "./nn";
import {HeatMap, reduceMatrix} from "./heatmap";
import {
  State,
  datasets,
  regDatasets,
  activations,
  problems,
  regularizations,
  getKeyFromValue,
  Problem
} from "./state";
import {Example2D, shuffle} from "./dataset";
import {AppendingLineChart} from "./linechart";
import {makeClientsFromXY} from "./fl/partition";
import {aggregateDeltas, addScaled, diffWeights} from "./fl/algorithms";
import {clipUpdate, addGaussianNoise} from "./fl/dp";
import {ServerAdam} from "./fl/optimizers";
import {ClientData, FLConfig, Weights} from "./fl/types";
import {kMeans, cosineSim, l2Dist2} from "./fl/cluster";
import * as d3 from 'd3';

let mainWidth;
let flClients: ClientData[] = null;
let flLastSeed = "";
let flLastSig = "";
let flServerAdam: ServerAdam = null;
let flLastAlgo = "";

// SCAFFOLD state
let flScaffoldC: Float32Array|null = null;          // server control variate c
let flScaffoldCi: (Float32Array|null)[] = [];       // per-client control variates c_i

// Cluster
let flClusterWeights: Float32Array[] | null = null;      // per-cluster model flats
let flClientCluster: number[] = [];                      // client index -> cluster id
let flLastClientDelta: (Float32Array|null)[] = [];       // last delta per client
let flClusterCount = 1;
let flRoundsSinceCluster = 0;

// Add after existing global variables
let flMetrics = {
  participationRates: [],
  commCosts: [],
  clientLosses: [],
  convergenceRates: [],
  fairnessMetrics: [],
  dpPrivacyBudget: 0
};

let flCharts: {[key: string]: AppendingLineChart} = {};

// More scrolling
d3.select(".more button").on("click", function() {
  let position = 800;
  d3.transition()
    .duration(1000)
    .tween("scroll", scrollTween(position));
});

function scrollTween(offset) {
  return function() {
    let i = d3.interpolateNumber(window.pageYOffset ||
        document.documentElement.scrollTop, offset);
    return function(t) { scrollTo(0, i(t)); };
  };
}

const RECT_SIZE = 30;
const BIAS_SIZE = 5;
const NUM_SAMPLES_CLASSIFY = 500;
const NUM_SAMPLES_REGRESS = 1200;
const DENSITY = 100;

enum HoverType {
  BIAS, WEIGHT
}

interface InputFeature {
  f: (x: number, y: number) => number;
  label?: string;
}

let INPUTS: {[name: string]: InputFeature} = {
  "x": {f: (x, y) => x, label: "X_1"},
  "y": {f: (x, y) => y, label: "X_2"},
  "xSquared": {f: (x, y) => x * x, label: "X_1^2"},
  "ySquared": {f: (x, y) => y * y,  label: "X_2^2"},
  "xTimesY": {f: (x, y) => x * y, label: "X_1X_2"},
  "sinX": {f: (x, y) => Math.sin(x), label: "sin(X_1)"},
  "sinY": {f: (x, y) => Math.sin(y), label: "sin(X_2)"},
};

let HIDABLE_CONTROLS = [
  ["Show test data", "showTestData"],
  ["Discretize output", "discretize"],
  ["Play button", "playButton"],
  ["Step button", "stepButton"],
  ["Reset button", "resetButton"],
  ["Learning rate", "learningRate"],
  ["Activation", "activation"],
  ["Regularization", "regularization"],
  ["Regularization rate", "regularizationRate"],
  ["Problem type", "problem"],
  ["Which dataset", "dataset"],
  ["Ratio train data", "percTrainData"],
  ["Noise level", "noise"],
  ["Batch size", "batchSize"],
  ["# of hidden layers", "numHiddenLayers"],
];

class Player {
  private timerIndex = 0;
  private isPlaying = false;
  private callback: (isPlaying: boolean) => void = null;

  /** Plays/pauses the player. */
  playOrPause() {
    if (this.isPlaying) {
      this.isPlaying = false;
      this.pause();
    } else {
      this.isPlaying = true;
      if (iter === 0) {
        simulationStarted();
      }
      this.play();
    }
  }

  onPlayPause(callback: (isPlaying: boolean) => void) {
    this.callback = callback;
  }

  play() {
    this.pause();
    this.isPlaying = true;
    if (this.callback) {
      this.callback(this.isPlaying);
    }
    this.start(this.timerIndex);
  }

  pause() {
    this.timerIndex++;
    this.isPlaying = false;
    if (this.callback) {
      this.callback(this.isPlaying);
    }
  }

  private start(localTimerIndex: number) {
    d3.timer(() => {
      if (localTimerIndex < this.timerIndex) {
        return true;  // Done.
      }
      oneStep();
      return false;  // Not done.
    }, 0);
  }
}

let state = State.deserializeState();

// Filter out inputs that are hidden.
state.getHiddenProps().forEach(prop => {
  if (prop in INPUTS) {
    delete INPUTS[prop];
  }
});

let boundary: {[id: string]: number[][]} = {};
let selectedNodeId: string = null;
// Plot the heatmap.
let xDomain: [number, number] = [-6, 6];
let heatMap =
    new HeatMap(300, DENSITY, xDomain, xDomain, d3.select("#heatmap"),
        {showAxes: true});
let linkWidthScale = d3.scale.linear()
  .domain([0, 5])
  .range([1, 10])
  .clamp(true);
let colorScale = d3.scale.linear<string, number>()
                     .domain([-1, 0, 1])
                     .range(["#f59322", "#e8eaeb", "#0877bd"])
                     .clamp(true);
let iter = 0;
let trainData: Example2D[] = [];
let testData: Example2D[] = [];
let network: nn.Node[][] = null;
let lossTrain = 0;
let lossTest = 0;
let player = new Player();
let lineChart = new AppendingLineChart(d3.select("#linechart"),
    ["#777", "black"]);

function getElNum(id: string, def: number): number {
  var el = document.getElementById(id) as HTMLInputElement;
  if (!el) return def;
  var v = parseFloat(el.value);
  return isNaN(v) ? def : v;
}
function getElInt(id: string, def: number): number {
  var n = getElNum(id, def);
  return Math.round(n);
}
function getElChecked(id: string): boolean {
  var el = document.getElementById(id) as HTMLInputElement;
  return !!(el && el.checked);
}
function getElSel(id: string, def: string): string {
  var el = document.getElementById(id) as HTMLSelectElement;
  return el && el.value ? el.value : def;
}
function isFLEnabled(): boolean {
  return getElChecked("fl-enabled");
}

function readFLConfig(): FLConfig {
  const algo = getElSel("fl-algo", "FedAvg") as any;
  const cfg: FLConfig = {
    algo,
    numClients: getElInt("fl-numClients", 30),
    clientFrac: Math.max(0.05, Math.min(1, +getElNum("fl-clientFrac", 0.2))),
    localEpochs: getElInt("fl-localEpochs", 1),
    batchSize: getElInt("batchSize", 16),
    clientLR: getElNum("fl-clientLR", 0.03),
    clientOptimizer: "sgd",
    weightedAggregation: getElChecked("fl-weighted"),
    clientDropout: Math.max(0, Math.min(0.9, +getElNum("fl-dropout", 0))),
    iidAlpha: getElNum("fl-alpha", 10),
    mu: getElNum("fl-mu", 0.1),
    serverLR: getElNum("fl-serverLR", 0.01),
    beta1: getElNum("fl-beta1", 0.9),
    beta2: getElNum("fl-beta2", 0.999),
    eps: getElNum("fl-eps", 1e-8),
    dpClipNorm: getElNum("fl-clip", 0),
    dpNoiseMult: getElNum("fl-noise", 0),
    dpClientLevel: getElChecked("fl-clientLevelDp"),
    clusteringEnabled: getElChecked("fl-clustered"),
    numClusters: Math.max(1, getElInt("fl-numClusters", 2)),
    reclusterEvery: Math.max(1, getElInt("fl-reclusterEvery", 5)),
    warmupRounds: Math.max(0, getElInt("fl-warmupRounds", 1)),
    similarityMetric: getElSel("fl-similarity", "cosine") as any,
    
  };
  return cfg;
}



function makeGUI() {
  // Clean up the FL control handlers - remove duplicates and organize properly
  d3.select("#fl-enabled").on("change", function() {
    const flControls = d3.select(".fl-controls");
    const flAdvanced = d3.select(".fl-advanced-controls");
    const flCluster = d3.select(".fl-cluster-controls");
    const flMetrics = d3.select("#fl-metrics-container");
    
    if (this.checked) {
      flControls.style("display", "block");
      initFLCharts();
    } else {
      flControls.style("display", "none");
      flAdvanced.style("display", "none");
      flCluster.style("display", "none");
      flMetrics.style("display", "none");
    }
    
    parametersChanged = true;
    userHasInteracted();
  });

  // FL Advanced Toggle Logic
  d3.select("#fl-advanced-toggle").on("change", function() {
    const flAdvanced = d3.select(".fl-advanced-controls");
    const flCluster = d3.select(".fl-cluster-controls");
    
    if (this.checked) {
      flAdvanced.style("display", "block");
      flCluster.style("display", "block");
    } else {
      flAdvanced.style("display", "none");
      flCluster.style("display", "none");
    }
    
    parametersChanged = true;
    userHasInteracted();
  });

  // Add missing Clustered FL toggle handler
  d3.select("#fl-clustered").on("change", function() {
    const flCluster = d3.select(".fl-cluster-controls");
    
    if (this.checked) {
      flCluster.style("display", "block");
    } else {
      flCluster.style("display", "none");
    }
    
    parametersChanged = true;
    userHasInteracted();
  });

  // Differential Privacy Controls Toggle
  d3.select("#fl-clientLevelDp").on("change", function() {
    const dpControls = d3.selectAll(".ui-fl-clip, .ui-fl-noise");
    
    if (this.checked) {
      dpControls.style("display", "block");
    } else {
      dpControls.style("display", "none");
    }
    
    parametersChanged = true;
    userHasInteracted();
  });
  
  d3.select("#reset-button").on("click", () => {
    reset();
    userHasInteracted();
    d3.select("#play-pause-button");
  });

  d3.select("#play-pause-button").on("click", function () {
    // Change the button's content.
    userHasInteracted();
    player.playOrPause();
  });

  player.onPlayPause(isPlaying => {
    d3.select("#play-pause-button").classed("playing", isPlaying);
  });

  d3.select("#next-step-button").on("click", () => {
    player.pause();
    userHasInteracted();
    if (iter === 0) {
      simulationStarted();
    }
    oneStep();
  });

  d3.select("#data-regen-button").on("click", () => {
    generateData();
    parametersChanged = true;
  });

  let dataThumbnails = d3.selectAll("canvas[data-dataset]");
  dataThumbnails.on("click", function() {
    let newDataset = datasets[this.dataset.dataset];
    if (newDataset === state.dataset) {
      return; // No-op.
    }
    state.dataset =  newDataset;
    dataThumbnails.classed("selected", false);
    d3.select(this).classed("selected", true);
    generateData();
    parametersChanged = true;
    reset();
  });

  let datasetKey = getKeyFromValue(datasets, state.dataset);
  // Select the dataset according to the current state.
  d3.select(`canvas[data-dataset=${datasetKey}]`)
    .classed("selected", true);

  let regDataThumbnails = d3.selectAll("canvas[data-regDataset]");
  regDataThumbnails.on("click", function() {
    let newDataset = regDatasets[this.dataset.regdataset];
    if (newDataset === state.regDataset) {
      return; // No-op.
    }
    state.regDataset =  newDataset;
    regDataThumbnails.classed("selected", false);
    d3.select(this).classed("selected", true);
    generateData();
    parametersChanged = true;
    reset();
  });

  let regDatasetKey = getKeyFromValue(regDatasets, state.regDataset);
  // Select the dataset according to the current state.
  d3.select(`canvas[data-regDataset=${regDatasetKey}]`)
    .classed("selected", true);

  d3.select("#add-layers").on("click", () => {
    if (state.numHiddenLayers >= 6) {
      return;
    }
    state.networkShape[state.numHiddenLayers] = 2;
    state.numHiddenLayers++;
    parametersChanged = true;
    reset();
  });

  d3.select("#remove-layers").on("click", () => {
    if (state.numHiddenLayers <= 0) {
      return;
    }
    state.numHiddenLayers--;
    state.networkShape.splice(state.numHiddenLayers);
    parametersChanged = true;
    reset();
  });

  let showTestData = d3.select("#show-test-data").on("change", function() {
    state.showTestData = this.checked;
    state.serialize();
    userHasInteracted();
    heatMap.updateTestPoints(state.showTestData ? testData : []);
  });
  // Check/uncheck the checkbox according to the current state.
  showTestData.property("checked", state.showTestData);

  let discretize = d3.select("#discretize").on("change", function() {
    state.discretize = this.checked;
    state.serialize();
    userHasInteracted();
    updateUI();
  });
  // Check/uncheck the checbox according to the current state.
  discretize.property("checked", state.discretize);

  let percTrain = d3.select("#percTrainData").on("input", function() {
    state.percTrainData = this.value;
    d3.select("label[for='percTrainData'] .value").text(this.value);
    generateData();
    parametersChanged = true;
    reset();
  });
  percTrain.property("value", state.percTrainData);
  d3.select("label[for='percTrainData'] .value").text(state.percTrainData);

  let noise = d3.select("#noise").on("input", function() {
    state.noise = this.value;
    d3.select("label[for='noise'] .value").text(this.value);
    generateData();
    parametersChanged = true;
    reset();
  });
  let currentMax = parseInt(noise.property("max"));
  if (state.noise > currentMax) {
    if (state.noise <= 80) {
      noise.property("max", state.noise);
    } else {
      state.noise = 50;
    }
  } else if (state.noise < 0) {
    state.noise = 0;
  }
  noise.property("value", state.noise);
  d3.select("label[for='noise'] .value").text(state.noise);

  let batchSize = d3.select("#batchSize").on("input", function() {
    state.batchSize = this.value;
    d3.select("label[for='batchSize'] .value").text(this.value);
    parametersChanged = true;
    reset();
  });
  batchSize.property("value", state.batchSize);
  d3.select("label[for='batchSize'] .value").text(state.batchSize);

  let activationDropdown = d3.select("#activations").on("change", function() {
    state.activation = activations[this.value];
    parametersChanged = true;
    reset();
  });
  activationDropdown.property("value",
      getKeyFromValue(activations, state.activation));

  let learningRate = d3.select("#learningRate").on("change", function() {
    state.learningRate = +this.value;
    state.serialize();
    userHasInteracted();
    parametersChanged = true;
  });
  learningRate.property("value", state.learningRate);

  let regularDropdown = d3.select("#regularizations").on("change",
      function() {
    state.regularization = regularizations[this.value];
    parametersChanged = true;
    reset();
  });
  regularDropdown.property("value",
      getKeyFromValue(regularizations, state.regularization));

  let regularRate = d3.select("#regularRate").on("change", function() {
    state.regularizationRate = +this.value;
    parametersChanged = true;
    reset();
  });
  regularRate.property("value", state.regularizationRate);

  let problem = d3.select("#problem").on("change", function() {
    state.problem = problems[this.value];
    generateData();
    drawDatasetThumbnails();
    parametersChanged = true;
    reset();
  });
  problem.property("value", getKeyFromValue(problems, state.problem));

  // Move all the bindMirror calls and algorithm-specific logic here
  function bindMirror(idRange: string, idSpan: string) {
    var el = document.getElementById(idRange) as HTMLInputElement;
    var sp = document.getElementById(idSpan) as HTMLElement;
    if (!el || !sp) return;
    var update = function(){ sp.textContent = el.value; };
    el.addEventListener("input", update);
    update();
  }
  
  bindMirror("fl-numClients", "fl-numClients-val");
  bindMirror("fl-clientFrac", "fl-clientFrac-val");
  bindMirror("fl-localEpochs", "fl-localEpochs-val");
  bindMirror("fl-clientLR", "fl-clientLR-val");
  bindMirror("fl-alpha", "fl-alpha-val");
  bindMirror("fl-dropout", "fl-dropout-val");
  bindMirror("fl-clip", "fl-clip-val");
  bindMirror("fl-noise", "fl-noise-val");
  bindMirror("fl-numClusters", "fl-numClusters-val");
  bindMirror("fl-reclusterEvery", "fl-reclusterEvery-val");
  bindMirror("fl-warmupRounds", "fl-warmupRounds-val");

  // Show/hide FedAdam/FedProx options
  var algoSel = document.getElementById("fl-algo") as HTMLSelectElement;
  function showAlgoOpts(){
    var v = algoSel ? algoSel.value : "FedAvg";
    var a = document.getElementById("fedadam-opts") as HTMLElement;
    var p = document.getElementById("fedprox-opts") as HTMLElement;
    if (a) a.style.display = (v === "FedAdam") ? "" : "none";
    if (p) p.style.display = (v === "FedProx") ? "" : "none";
  }
  if (algoSel) {
    algoSel.addEventListener("change", function(){ 
      flServerAdam = null; 
      flScaffoldC = null; 
      flScaffoldCi = []; 
      showAlgoOpts(); 
    });
    showAlgoOpts();
  }
  
  // Add scale to the gradient color map.
  let x = d3.scale.linear().domain([-1, 1]).range([0, 144]);
  let xAxis = d3.svg.axis()
    .scale(x)
    .orient("bottom")
    .tickValues([-1, 0, 1])
    .tickFormat(d3.format("d"));
  d3.select("#colormap g.core").append("g")
    .attr("class", "x axis")
    .attr("transform", "translate(0,10)")
    .call(xAxis);

  // Listen for css-responsive changes and redraw the svg network.
  window.addEventListener("resize", () => {
    let newWidth = document.querySelector("#main-part")
        .getBoundingClientRect().width;
    if (newWidth !== mainWidth) {
      mainWidth = newWidth;
      drawNetwork(network);
      updateUI(true);
    }
  });

  // Hide the text below the visualization depending on the URL.
  if (state.hideText) {
    d3.select("#article-text").style("display", "none");
    d3.select("div.more").style("display", "none");
    d3.select("header").style("display", "none");
  }
}

  function bindMirror(idRange: string, idSpan: string) {
    var el = document.getElementById(idRange) as HTMLInputElement;
    var sp = document.getElementById(idSpan) as HTMLElement;
    if (!el || !sp) return;
    var update = function(){ sp.textContent = el.value; };
    el.addEventListener("input", update);
      update();
    }
    bindMirror("fl-numClients", "fl-numClients-val");
    bindMirror("fl-clientFrac", "fl-clientFrac-val");
    bindMirror("fl-localEpochs", "fl-localEpochs-val");
    bindMirror("fl-clientLR", "fl-clientLR-val");
    bindMirror("fl-alpha", "fl-alpha-val");
    bindMirror("fl-dropout", "fl-dropout-val");
    bindMirror("fl-clip", "fl-clip-val");
    bindMirror("fl-noise", "fl-noise-val");
    bindMirror("fl-dropout", "fl-dropout-val");
    bindMirror("fl-clip", "fl-clip-val");
    bindMirror("fl-noise", "fl-noise-val");
    bindMirror("fl-numClusters", "fl-numClusters-val");
    bindMirror("fl-reclusterEvery", "fl-reclusterEvery-val");
    bindMirror("fl-warmupRounds", "fl-warmupRounds-val");
    bindMirror("fl-mu", "fl-mu-val");

    // Show/hide FedAdam/FedProx options
    var algoSel = document.getElementById("fl-algo") as HTMLSelectElement;
    function showAlgoOpts(){
      var v = algoSel ? algoSel.value : "FedAvg";
      var a = document.getElementById("fedadam-opts") as HTMLElement;
      var p = document.getElementById("fedprox-opts") as HTMLElement;
      if (a) a.style.display = (v === "FedAdam") ? "" : "none";
      if (p) p.style.display = (v === "FedProx") ? "" : "none";
    }
    if (algoSel) {
      algoSel.addEventListener("change", function(){ 
        flServerAdam = null; 
        flScaffoldC = null; 
        flScaffoldCi = []; 
        showAlgoOpts(); 
      });
      showAlgoOpts();
    }
  
  d3.select("#fl-enabled").on("change", function() {
    const flControls = d3.select(".fl-controls");
    const flAdvanced = d3.select(".fl-advanced-controls");
    
    if (this.checked) {
      flControls.style("display", "block");
    } else {
      flControls.style("display", "none");
      flAdvanced.style("display", "none");
    }
    
    parametersChanged = true;
    userHasInteracted();
  });

  // FL Advanced Toggle Logic
  d3.select("#fl-advanced-toggle").on("change", function() {
    const flAdvanced = d3.select(".fl-advanced-controls");
    const flCluster = d3.select(".fl-cluster-controls");
    
    if (this.checked) {
      flAdvanced.style("display", "block");
      flCluster.style("display", "block");
    } else {
      flAdvanced.style("display", "none");
      flCluster.style("display", "none");
    }
    
    parametersChanged = true;
    userHasInteracted();
  });

  // Differential Privacy Controls Toggle
  d3.select("#fl-clientLevelDp").on("change", function() {
    const dpControls = d3.selectAll(".ui-fl-clip, .ui-fl-noise");
    
    if (this.checked) {
      dpControls.style("display", "block");
    } else {
      dpControls.style("display", "none");
    }
    
    parametersChanged = true;
    userHasInteracted();
  });
  
  // Add scale to the gradient color map.
  let x = d3.scale.linear().domain([-1, 1]).range([0, 144]);
  let xAxis = d3.svg.axis()
    .scale(x)
    .orient("bottom")
    .tickValues([-1, 0, 1])
    .tickFormat(d3.format("d"));
  d3.select("#colormap g.core").append("g")
    .attr("class", "x axis")
    .attr("transform", "translate(0,10)")
    .call(xAxis);

  // Listen for css-responsive changes and redraw the svg network.

  window.addEventListener("resize", () => {
    let newWidth = document.querySelector("#main-part")
        .getBoundingClientRect().width;
    if (newWidth !== mainWidth) {
      mainWidth = newWidth;
      drawNetwork(network);
      updateUI(true);
    }
  });

  // Hide the text below the visualization depending on the URL.
  if (state.hideText) {
    d3.select("#article-text").style("display", "none");
    d3.select("div.more").style("display", "none");
    d3.select("header").style("display", "none");
  };

function updateBiasesUI(network: nn.Node[][]) {
  nn.forEachNode(network, true, node => {
    d3.select(`rect#bias-${node.id}`).style("fill", colorScale(node.bias));
  });
}

function updateWeightsUI(network: nn.Node[][], container) {
  for (let layerIdx = 1; layerIdx < network.length; layerIdx++) {
    let currentLayer = network[layerIdx];
    // Update all the nodes in this layer.
    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i];
      for (let j = 0; j < node.inputLinks.length; j++) {
        let link = node.inputLinks[j];
        container.select(`#link${link.source.id}-${link.dest.id}`)
            .style({
              "stroke-dashoffset": -iter / 3,
              "stroke-width": linkWidthScale(Math.abs(link.weight)),
              "stroke": colorScale(link.weight)
            })
            .datum(link);
      }
    }
  }
}

function drawNode(cx: number, cy: number, nodeId: string, isInput: boolean,
    container, node?: nn.Node) {
  let x = cx - RECT_SIZE / 2;
  let y = cy - RECT_SIZE / 2;

  let nodeGroup = container.append("g")
    .attr({
      "class": "node",
      "id": `node${nodeId}`,
      "transform": `translate(${x},${y})`
    });

  // Draw the main rectangle.
  nodeGroup.append("rect")
    .attr({
      x: 0,
      y: 0,
      width: RECT_SIZE,
      height: RECT_SIZE,
    });
  let activeOrNotClass = state[nodeId] ? "active" : "inactive";
  if (isInput) {
    let label = INPUTS[nodeId].label != null ?
        INPUTS[nodeId].label : nodeId;
    // Draw the input label.
    let text = nodeGroup.append("text").attr({
      class: "main-label",
      x: -10,
      y: RECT_SIZE / 2, "text-anchor": "end"
    });
    if (/[_^]/.test(label)) {
      let myRe = /(.*?)([_^])(.)/g;
      let myArray;
      let lastIndex;
      while ((myArray = myRe.exec(label)) != null) {
        lastIndex = myRe.lastIndex;
        let prefix = myArray[1];
        let sep = myArray[2];
        let suffix = myArray[3];
        if (prefix) {
          text.append("tspan").text(prefix);
        }
        text.append("tspan")
        .attr("baseline-shift", sep === "_" ? "sub" : "super")
        .style("font-size", "9px")
        .text(suffix);
      }
      if (label.substring(lastIndex)) {
        text.append("tspan").text(label.substring(lastIndex));
      }
    } else {
      text.append("tspan").text(label);
    }
    nodeGroup.classed(activeOrNotClass, true);
  }
  if (!isInput) {
    // Draw the node's bias.
    nodeGroup.append("rect")
      .attr({
        id: `bias-${nodeId}`,
        x: -BIAS_SIZE - 2,
        y: RECT_SIZE - BIAS_SIZE + 3,
        width: BIAS_SIZE,
        height: BIAS_SIZE,
      }).on("mouseenter", function() {
        updateHoverCard(HoverType.BIAS, node, d3.mouse(container.node()));
      }).on("mouseleave", function() {
        updateHoverCard(null);
      });
  }

  // Draw the node's canvas.
  let div = d3.select("#network").insert("div", ":first-child")
    .attr({
      "id": `canvas-${nodeId}`,
      "class": "canvas"
    })
    .style({
      position: "absolute",
      left: `${x + 3}px`,
      top: `${y + 3}px`
    })
    .on("mouseenter", function() {
      selectedNodeId = nodeId;
      div.classed("hovered", true);
      nodeGroup.classed("hovered", true);
      updateDecisionBoundary(network, false);
      heatMap.updateBackground(boundary[nodeId], state.discretize);
    })
    .on("mouseleave", function() {
      selectedNodeId = null;
      div.classed("hovered", false);
      nodeGroup.classed("hovered", false);
      updateDecisionBoundary(network, false);
      heatMap.updateBackground(boundary[nn.getOutputNode(network).id],
          state.discretize);
    });
  if (isInput) {
    div.on("click", function() {
      state[nodeId] = !state[nodeId];
      parametersChanged = true;
      reset();
    });
    div.style("cursor", "pointer");
  }
  if (isInput) {
    div.classed(activeOrNotClass, true);
  }
  let nodeHeatMap = new HeatMap(RECT_SIZE, DENSITY / 10, xDomain,
      xDomain, div, {noSvg: true});
  div.datum({heatmap: nodeHeatMap, id: nodeId});

}

// Draw network
function drawNetwork(network: nn.Node[][]): void {
  let svg = d3.select("#svg");
  // Remove all svg elements.
  svg.select("g.core").remove();
  // Remove all div elements.
  d3.select("#network").selectAll("div.canvas").remove();
  d3.select("#network").selectAll("div.plus-minus-neurons").remove();

  // Get the width of the svg container.
  let padding = 3;
  let co = d3.select(".column.output").node() as HTMLDivElement;
  let cf = d3.select(".column.features").node() as HTMLDivElement;
  let width = co.offsetLeft - cf.offsetLeft;
  svg.attr("width", width);

  // Map of all node coordinates.
  let node2coord: {[id: string]: {cx: number, cy: number}} = {};
  let container = svg.append("g")
    .classed("core", true)
    .attr("transform", `translate(${padding},${padding})`);
  // Draw the network layer by layer.
  let numLayers = network.length;
  let featureWidth = 118;
  let layerScale = d3.scale.ordinal<number, number>()
      .domain(d3.range(1, numLayers - 1))
      .rangePoints([featureWidth, width - RECT_SIZE], 0.7);
  let nodeIndexScale = (nodeIndex: number) => nodeIndex * (RECT_SIZE + 25);


  let calloutThumb = d3.select(".callout.thumbnail").style("display", "none");
  let calloutWeights = d3.select(".callout.weights").style("display", "none");
  let idWithCallout = null;
  let targetIdWithCallout = null;

  // Draw the input layer separately.
  let cx = RECT_SIZE / 2 + 50;
  let nodeIds = Object.keys(INPUTS);
  let maxY = nodeIndexScale(nodeIds.length);
  nodeIds.forEach((nodeId, i) => {
    let cy = nodeIndexScale(i) + RECT_SIZE / 2;
    node2coord[nodeId] = {cx, cy};
    drawNode(cx, cy, nodeId, true, container);
  });

  // Draw the intermediate layers.
  for (let layerIdx = 1; layerIdx < numLayers - 1; layerIdx++) {
    let numNodes = network[layerIdx].length;
    let cx = layerScale(layerIdx) + RECT_SIZE / 2;
    maxY = Math.max(maxY, nodeIndexScale(numNodes));
    addPlusMinusControl(layerScale(layerIdx), layerIdx);
    for (let i = 0; i < numNodes; i++) {
      let node = network[layerIdx][i];
      let cy = nodeIndexScale(i) + RECT_SIZE / 2;
      node2coord[node.id] = {cx, cy};
      drawNode(cx, cy, node.id, false, container, node);

      // Show callout to thumbnails.
      let numNodes = network[layerIdx].length;
      let nextNumNodes = network[layerIdx + 1].length;
      if (idWithCallout == null &&
          i === numNodes - 1 &&
          nextNumNodes <= numNodes) {
        calloutThumb.style({
          display: null,
          top: `${20 + 3 + cy}px`,
          left: `${cx}px`
        });
        idWithCallout = node.id;
      }

      // Draw links.
      for (let j = 0; j < node.inputLinks.length; j++) {
        let link = node.inputLinks[j];
        let path: SVGPathElement = drawLink(link, node2coord, network,
            container, j === 0, j, node.inputLinks.length).node() as any;
        // Show callout to weights.
        let prevLayer = network[layerIdx - 1];
        let lastNodePrevLayer = prevLayer[prevLayer.length - 1];
        if (targetIdWithCallout == null &&
            i === numNodes - 1 &&
            link.source.id === lastNodePrevLayer.id &&
            (link.source.id !== idWithCallout || numLayers <= 5) &&
            link.dest.id !== idWithCallout &&
            prevLayer.length >= numNodes) {
          let midPoint = path.getPointAtLength(path.getTotalLength() * 0.7);
          calloutWeights.style({
            display: null,
            top: `${midPoint.y + 5}px`,
            left: `${midPoint.x + 3}px`
          });
          targetIdWithCallout = link.dest.id;
        }
      }
    }
  }

  // Draw the output node separately.
  cx = width + RECT_SIZE / 2;
  let node = network[numLayers - 1][0];
  let cy = nodeIndexScale(0) + RECT_SIZE / 2;
  node2coord[node.id] = {cx, cy};
  // Draw links.
  for (let i = 0; i < node.inputLinks.length; i++) {
    let link = node.inputLinks[i];
    drawLink(link, node2coord, network, container, i === 0, i,
        node.inputLinks.length);
  }
  // Adjust the height of the svg.
  svg.attr("height", maxY);

  // Adjust the height of the features column.
  let height = Math.max(
    getRelativeHeight(calloutThumb),
    getRelativeHeight(calloutWeights),
    getRelativeHeight(d3.select("#network"))
  );
  d3.select(".column.features").style("height", height + "px");
}

function getRelativeHeight(selection) {
  let node = selection.node() as HTMLAnchorElement;
  return node.offsetHeight + node.offsetTop;
}

function addPlusMinusControl(x: number, layerIdx: number) {
  let div = d3.select("#network").append("div")
    .classed("plus-minus-neurons", true)
    .style("left", `${x - 10}px`);

  let i = layerIdx - 1;
  let firstRow = div.append("div").attr("class", `ui-numNodes${layerIdx}`);
  firstRow.append("button")
      .attr("class", "mdl-button mdl-js-button mdl-button--icon")
      .on("click", () => {
        let numNeurons = state.networkShape[i];
        if (numNeurons >= 8) {
          return;
        }
        state.networkShape[i]++;
        parametersChanged = true;
        reset();
      })
    .append("i")
      .attr("class", "material-icons")
      .text("add");

  firstRow.append("button")
      .attr("class", "mdl-button mdl-js-button mdl-button--icon")
      .on("click", () => {
        let numNeurons = state.networkShape[i];
        if (numNeurons <= 1) {
          return;
        }
        state.networkShape[i]--;
        parametersChanged = true;
        reset();
      })
    .append("i")
      .attr("class", "material-icons")
      .text("remove");

  let suffix = state.networkShape[i] > 1 ? "s" : "";
  div.append("div").text(
    state.networkShape[i] + " neuron" + suffix
  );
}

function updateHoverCard(type: HoverType, nodeOrLink?: nn.Node | nn.Link,
    coordinates?: [number, number]) {
  let hovercard = d3.select("#hovercard");
  if (type == null) {
    hovercard.style("display", "none");
    d3.select("#svg").on("click", null);
    return;
  }
  d3.select("#svg").on("click", () => {
    hovercard.select(".value").style("display", "none");
    let input = hovercard.select("input");
    input.style("display", null);
    input.on("input", function() {
      if (this.value != null && this.value !== "") {
        if (type === HoverType.WEIGHT) {
          (nodeOrLink as nn.Link).weight = +this.value;
        } else {
          (nodeOrLink as nn.Node).bias = +this.value;
        }
        updateUI();
      }
    });
    input.on("keypress", () => {
      if ((d3.event as any).keyCode === 13) {
        updateHoverCard(type, nodeOrLink, coordinates);
      }
    });
    (input.node() as HTMLInputElement).focus();
  });
  let value = (type === HoverType.WEIGHT) ?
    (nodeOrLink as nn.Link).weight :
    (nodeOrLink as nn.Node).bias;
  let name = (type === HoverType.WEIGHT) ? "Weight" : "Bias";
  hovercard.style({
    "left": `${coordinates[0] + 20}px`,
    "top": `${coordinates[1]}px`,
    "display": "block"
  });
  hovercard.select(".type").text(name);
  hovercard.select(".value")
    .style("display", null)
    .text(value.toPrecision(2));
  hovercard.select("input")
    .property("value", value.toPrecision(2))
    .style("display", "none");
}

function drawLink(
    input: nn.Link, node2coord: {[id: string]: {cx: number, cy: number}},
    network: nn.Node[][], container,
    isFirst: boolean, index: number, length: number) {
  let line = container.insert("path", ":first-child");
  let source = node2coord[input.source.id];
  let dest = node2coord[input.dest.id];
  let datum = {
    source: {
      y: source.cx + RECT_SIZE / 2 + 2,
      x: source.cy
    },
    target: {
      y: dest.cx - RECT_SIZE / 2,
      x: dest.cy + ((index - (length - 1) / 2) / length) * 12
    }
  };
  let diagonal = d3.svg.diagonal().projection(d => [d.y, d.x]);
  line.attr({
    "marker-start": "url(#markerArrow)",
    class: "link",
    id: "link" + input.source.id + "-" + input.dest.id,
    d: diagonal(datum, 0)
  });

  // Add an invisible thick link that will be used for
  // showing the weight value on hover.
  container.append("path")
    .attr("d", diagonal(datum, 0))
    .attr("class", "link-hover")
    .on("mouseenter", function() {
      updateHoverCard(HoverType.WEIGHT, input, d3.mouse(this));
    }).on("mouseleave", function() {
      updateHoverCard(null);
    });
  return line;
}

/**
 * Given a neural network, it asks the network for the output (prediction)
 * of every node in the network using inputs sampled on a square grid.
 * It returns a map where each key is the node ID and the value is a square
 * matrix of the outputs of the network for each input in the grid respectively.
 */
function updateDecisionBoundary(network: nn.Node[][], firstTime: boolean) {
  if (firstTime) {
    boundary = {};
    nn.forEachNode(network, true, node => {
      boundary[node.id] = new Array(DENSITY);
    });
    // Go through all predefined inputs.
    for (let nodeId in INPUTS) {
      boundary[nodeId] = new Array(DENSITY);
    }
  }
  let xScale = d3.scale.linear().domain([0, DENSITY - 1]).range(xDomain);
  let yScale = d3.scale.linear().domain([DENSITY - 1, 0]).range(xDomain);

  let i = 0, j = 0;
  for (i = 0; i < DENSITY; i++) {
    if (firstTime) {
      nn.forEachNode(network, true, node => {
        boundary[node.id][i] = new Array(DENSITY);
      });
      // Go through all predefined inputs.
      for (let nodeId in INPUTS) {
        boundary[nodeId][i] = new Array(DENSITY);
      }
    }
    for (j = 0; j < DENSITY; j++) {
      // 1 for points inside the circle, and 0 for points outside the circle.
      let x = xScale(i);
      let y = yScale(j);
      let input = constructInput(x, y);
      nn.forwardProp(network, input);
      nn.forEachNode(network, true, node => {
        boundary[node.id][i][j] = node.output;
      });
      if (firstTime) {
        // Go through all predefined inputs.
        for (let nodeId in INPUTS) {
          boundary[nodeId][i][j] = INPUTS[nodeId].f(x, y);
        }
      }
    }
  }
}

function getLoss(network: nn.Node[][], dataPoints: Example2D[]): number {
  let loss = 0;
  for (let i = 0; i < dataPoints.length; i++) {
    let dataPoint = dataPoints[i];
    let input = constructInput(dataPoint.x, dataPoint.y);
    let output = nn.forwardProp(network, input);
    loss += nn.Errors.SQUARE.error(output, dataPoint.label);
  }
  return loss / dataPoints.length;
}

function updateUI(firstStep = false) {
  // Update the links visually.
  updateWeightsUI(network, d3.select("g.core"));
  // Update the bias values visually.
  updateBiasesUI(network);
  // Get the decision boundary of the network.
  updateDecisionBoundary(network, firstStep);
  let selectedId = selectedNodeId != null ?
      selectedNodeId : nn.getOutputNode(network).id;
  heatMap.updateBackground(boundary[selectedId], state.discretize);

  // Update all decision boundaries.
  d3.select("#network").selectAll("div.canvas")
      .each(function(data: {heatmap: HeatMap, id: string}) {
    data.heatmap.updateBackground(reduceMatrix(boundary[data.id], 10),
        state.discretize);
  });

  function zeroPad(n: number): string {
    let pad = "000000";
    return (pad + n).slice(-pad.length);
  }

  function addCommas(s: string): string {
    return s.replace(/\B(?=(\d{3})+(?!\d))/g, ",");
  }

  function humanReadable(n: number): string {
    return n.toFixed(3);
  }

  // Update loss and iteration number.
  d3.select("#loss-train").text(humanReadable(lossTrain));
  d3.select("#loss-test").text(humanReadable(lossTest));
  d3.select("#iter-number").text(addCommas(zeroPad(iter)));
  lineChart.addDataPoint([lossTrain, lossTest]);
}

function constructInputIds(): string[] {
  let result: string[] = [];
  for (let inputName in INPUTS) {
    if (state[inputName]) {
      result.push(inputName);
    }
  }
  return result;
}

function constructInput(x: number, y: number): number[] {
  let input: number[] = [];
  for (let inputName in INPUTS) {
    if (state[inputName]) {
      input.push(INPUTS[inputName].f(x, y));
    }
  }
  return input;
}


function rebuildFLClientsIfNeeded(cfg: FLConfig): void {
  // Rebuild when seed changes or partition knobs change.
  var sig = state.seed + "|" + cfg.numClients + "|" + state.batchSize + "|" + cfg.iidAlpha + "|" + state.problem;
  if (flClients && flLastSeed === state.seed && flLastSig === sig) return;

  // Only classification is supported here (2 classes in the Playground).
  var X: number[][] = [];
  var Y: number[] = [];
  for (var i = 0; i < trainData.length; i++) {
    X.push([trainData[i].x, trainData[i].y]);
    Y.push(trainData[i].label > 0 ? 1 : 0);
  }
  flClients = makeClientsFromXY(X, Y, 2, cfg.numClients, state.batchSize, cfg.iidAlpha);
  flLastSeed = state.seed;
  flLastSig = sig;

  // Reset SCAFFOLD variates on client topology changes
  flScaffoldC = null;
  flScaffoldCi = [];
  for (let i2 = 0; i2 < flClients.length; i2++) flScaffoldCi.push(null);

  // Reset clustered FL (avoid Array.fill for older TS targets)
  var arr0 = new Array(flClients.length);
  for (var a0 = 0; a0 < arr0.length; a0++) arr0[a0] = 0;
  flClientCluster = arr0 as number[];

  var arrN = new Array(flClients.length);
  for (var aN = 0; aN < arrN.length; aN++) arrN[aN] = null;
  flLastClientDelta = arrN as (Float32Array|null)[];

  flClusterWeights = null;
  flClusterCount = 1;
  flRoundsSinceCluster = 0;
}

function nnFlattenWeights(): Float32Array {
  // Order: biases of non-input layers, then each link weight.
  var arr: number[] = [];
  for (var layerIdx = 1; layerIdx < network.length; layerIdx++) {
    var layer = network[layerIdx];
    for (var i = 0; i < layer.length; i++) arr.push(layer[i].bias);
    for (var i2 = 0; i2 < layer.length; i2++) {
      var node = layer[i2];
      for (var j = 0; j < node.inputLinks.length; j++) arr.push(node.inputLinks[j].weight);
    }
  }
  var out = new Float32Array(arr.length);
  for (var k = 0; k < arr.length; k++) out[k] = arr[k];
  return out;
}

function nnSetWeightsFromFlat(buf: Float32Array): void {
  var idx = 0;
  for (var layerIdx = 1; layerIdx < network.length; layerIdx++) {
    var layer = network[layerIdx];
    for (var i = 0; i < layer.length; i++) { layer[i].bias = buf[idx++]; }
    for (var i2 = 0; i2 < layer.length; i2++) {
      var node = layer[i2];
      for (var j = 0; j < node.inputLinks.length; j++) node.inputLinks[j].weight = buf[idx++];
    }
  }
}

function nnCloneWeights(): Weights {
  var f = nnFlattenWeights();
  var copy = new Float32Array(f.length);
  for (var i = 0; i < f.length; i++) copy[i] = f[i];
  return [copy]; // Weights = Float32Array[]
}

function nnSetWeights(w: Weights): void {
  nnSetWeightsFromFlat(w[0]);
}


// Helpers for clustered FL
// ...existing code...
function ensureClusterState(cfg: FLConfig): void {
  const K = Math.max(1, cfg.numClusters || 1);
  const base = nnFlattenWeights();
  if (!flClusterWeights || flClusterWeights.length !== K || flClusterWeights[0].length !== base.length) {
    flClusterWeights = [];
    for (let c = 0; c < K; c++) {
      const w = new Float32Array(base.length);
      w.set(base);
      flClusterWeights.push(w);
    }
  }
  if (flClientCluster.length !== flClients.length) {
    var a = new Array(flClients.length);
    for (var i = 0; i < a.length; i++) a[i] = 0;
    flClientCluster = a as number[];
  }
  if (flLastClientDelta.length !== flClients.length) {
    var b = new Array(flClients.length);
    for (var j = 0; j < b.length; j++) b[j] = null;
    flLastClientDelta = b as (Float32Array|null)[];
  }
  flClusterCount = K;
}

function averageClusterWeightsBySize(): Float32Array {
  if (!flClusterWeights) return nnFlattenWeights();
  var a = new Array(flClients.length);
  for (var i = 0; i < a.length; i++) a[i] = 0;
  const counts = a as number[];
  for (const c of flClientCluster) if (c >= 0 && c < counts.length) counts[c]++;
  let total = counts.reduce((a,b)=>a+b,0) || 1;
  const d = flClusterWeights[0].length;
  const out = new Float32Array(d);
  for (let c = 0; c < flClusterWeights.length; c++) {
    const w = flClusterWeights[c];
    const wgt = counts[c] / total;
    for (let i = 0; i < d; i++) out[i] += wgt * w[i];
  }
  return out;
}

function reclusterIfNeeded(cfg: FLConfig): void {
  if (!cfg.clusteringEnabled || (cfg.numClusters || 1) <= 1) return;
  if (flRoundsSinceCluster < (cfg.reclusterEvery || 5)) return;
  if ((iter + 1) < (cfg.warmupRounds || 0)) return;

  const K = Math.max(1, cfg.numClusters || 1);
  // Collect available deltas
  const vectors: Float32Array[] = [];
  const owners: number[] = [];
  for (let i = 0; i < flLastClientDelta.length; i++) {
    const v = flLastClientDelta[i];
    if (v) {
      vectors.push(v);
      owners.push(i);
    }
  }
  if (vectors.length === 0) { flRoundsSinceCluster = 0; return; }

  const {assignments, centroids} = kMeans(vectors, Math.min(K, vectors.length), (cfg.similarityMetric || "cosine") as any, 20, 42);

  // Build full assignment: use k-means for owners; others keep nearest centroid if they have delta, else keep prior.
  const newAssign = flClientCluster.slice();
  for (let i = 0; i < owners.length; i++) {
    newAssign[owners[i]] = assignments[i];
  }
  // If kMeans returned fewer centroids (vectors < K), extend by duplicating nearest
  while (centroids.length < K) centroids.push(centroids[centroids.length - 1]);

  for (let i = 0; i < newAssign.length; i++) {
    if (owners.indexOf(i) !== -1) continue;
    const v = flLastClientDelta[i];
    if (!v) continue;
    // Assign to nearest centroid by selected metric
    let best = 0;
    let bestScore = Number.POSITIVE_INFINITY;
    for (let c = 0; c < K; c++) {
      let score: number;
      if ((cfg.similarityMetric || "cosine") === "cosine") {
        // distance = 1 - cosine
        const sim = cosineSim(v, centroids[c]);
        score = 1 - sim;
      } else {
        score = l2Dist2(v, centroids[c]);
      }
      if (score < bestScore) { bestScore = score; best = c; }
    }
    newAssign[i] = best;
  }

  flClientCluster = newAssign;
  flRoundsSinceCluster = 0;
}

// ...existing code...

function initFLCharts() {
  if (!isFLEnabled()) return;
  
  // Show FL metrics container
  d3.select("#fl-metrics-container").style("display", "block");
  
  // Clear existing charts first
  d3.select("#participation-chart").selectAll("*").remove();
  d3.select("#comm-chart").selectAll("*").remove();
  d3.select("#client-loss-chart").selectAll("*").remove();
  d3.select("#convergence-chart").selectAll("*").remove();
  
  // Initialize charts with proper sizing
  flCharts.participation = new AppendingLineChart(
    d3.select("#participation-chart"),
    ["#2196F3", "#4CAF50"], // Blue for selected, green for participated
  );
  
  flCharts.communication = new AppendingLineChart(
    d3.select("#comm-chart"),
    ["#FF9800"], // Orange for comm cost
  );
  
  flCharts.clientLoss = new AppendingLineChart(
    d3.select("#client-loss-chart"),
    ["#F44336", "#9C27B0", "#607D8B"], // Red for max, purple for mean, blue-grey for min
  );
  
  flCharts.convergence = new AppendingLineChart(
    d3.select("#convergence-chart"),
    ["#795548"], // Brown for convergence rate
  );
}

// ...existing code...
function updateFLCharts() {
  if (!isFLEnabled() || Object.keys(flCharts).length === 0) return;
  
  // Update participation chart
  if (flMetrics.participationRates.length > 0) {
    const latest = flMetrics.participationRates[flMetrics.participationRates.length - 1];
    flCharts.participation.addDataPoint(latest);
  }
  
  // Update communication chart
  if (flMetrics.commCosts.length > 0) {
    const latest = flMetrics.commCosts[flMetrics.commCosts.length - 1];
    flCharts.communication.addDataPoint(latest);
  }
  
  // Update client loss distribution chart
  if (flMetrics.clientLosses.length > 0) {
    const latest = flMetrics.clientLosses[flMetrics.clientLosses.length - 1];
    flCharts.clientLoss.addDataPoint(latest);
  }
  
  // Update convergence chart
  if (flMetrics.convergenceRates.length > 0) {
    const latest = flMetrics.convergenceRates[flMetrics.convergenceRates.length - 1];
    flCharts.convergence.addDataPoint(latest);
  }
}

function oneStep(): void {
  if (isFLEnabled() && state.problem === Problem.CLASSIFICATION) {
    oneStepFL();
  } else {
    oneStepSGD();
  }
}
// ...existing code...
function oneStepFL(): void {
  var cfg = readFLConfig();
  if (flLastAlgo !== cfg.algo) { 
    flServerAdam = null; 
    flScaffoldC = null; 
    flScaffoldCi = []; 
    flLastAlgo = cfg.algo; 
  }

  rebuildFLClientsIfNeeded(cfg);

  // Clustered FL path (FedAvg/FedProx only)
  var clustered = !!cfg.clusteringEnabled && (cfg.numClusters || 1) > 1 && (cfg.algo === "FedAvg" || cfg.algo === "FedProx");
  if (clustered) {
    ensureClusterState(cfg);

    var K = flClusterCount;
    var roundStartTime = performance.now();

    // Sample clients
    var k = Math.max(1, Math.round((cfg.clientFrac || 0.2) * flClients.length));
    var shuffled = flClients.slice(0);
    shuffled.sort(function(){ return Math.random() - 0.5; });
    var dropout = (cfg.clientDropout !== undefined && cfg.clientDropout !== null) ? cfg.clientDropout : 0;
    var roundClients: ClientData[] = [];
    var selectedClients = 0;
    for (var i = 0; i < k && i < shuffled.length; i++) {
      selectedClients++;
      if (Math.random() > dropout) roundClients.push(shuffled[i]);
    }
    if (roundClients.length === 0 && shuffled.length > 0) roundClients.push(shuffled[0]);

    var participationRate = roundClients.length / Math.max(1, selectedClients);
    var clientParticipationRate = selectedClients / Math.max(1, flClients.length);

    // Per-cluster delta buckets
    var perCluster: {delta: Weights; weight: number; clientLoss: number}[][] = [];
    for (var ci0 = 0; ci0 < K; ci0++) perCluster.push([]);

    var clientLossValues: number[] = [];

    // Train each selected client against its cluster's model
    for (var rc = 0; rc < roundClients.length; rc++) {
      var client = roundClients[rc];
      var clientIdx = flClients.indexOf(client);
      var cid = Math.max(0, Math.min(K - 1, (flClientCluster[clientIdx] || 0)));

      // Set local weights to cluster weights
      var w0c = flClusterWeights![cid];
      var wLocalFlat = new Float32Array(w0c.length);
      for (var wi0 = 0; wi0 < w0c.length; wi0++) wLocalFlat[wi0] = w0c[wi0];
      nnSetWeightsFromFlat(wLocalFlat);

      // Local training
      for (var e = 0; e < (cfg.localEpochs || 1); e++) {
        for (var b = 0; b < client.batches.length; b++) {
          var batch = client.batches[b];
          for (var t = 0; t < batch.x.length; t++) {
            var xy = batch.x[t];
            var input = constructInput(xy[0], xy[1]);
            var target = batch.y[t] > 0 ? 1 : -1;
            nn.forwardProp(network, input);
            nn.backProp(network, target, nn.Errors.SQUARE);
          }

          // FedProx proximal term vs cluster's w0c
          if (cfg.algo === "FedProx" && cfg.mu && cfg.mu > 0) {
            var idx = 0;
            for (var layerIdx = 1; layerIdx < network.length; layerIdx++) {
              var layer = network[layerIdx];
              for (var i3 = 0; i3 < layer.length; i3++) {
                var node = layer[i3];
                var gBias = cfg.mu * (node.bias - w0c[idx++]);
                node.accInputDer += gBias;
                node.numAccumulatedDers += 1;
              }
              for (var i4 = 0; i4 < layer.length; i4++) {
                var node2 = layer[i4];
                for (var j = 0; j < node2.inputLinks.length; j++) {
                  var link = node2.inputLinks[j];
                  var gW = cfg.mu * (link.weight - w0c[idx++]);
                  link.accErrorDer += gW;
                  link.numAccumulatedDers += 1;
                }
              }
            }
          }

          nn.updateWeights(network, cfg.clientLR || 0.03, state.regularizationRate);
        }
      }

      // Client final loss
      var clientEndLoss = 0;
      var clientEndSamples = 0;
      for (var b2 = 0; b2 < client.batches.length; b2++) {
        var batch2 = client.batches[b2];
        for (var t2 = 0; t2 < batch2.x.length; t2++) {
          var xy2 = batch2.x[t2];
          var input2 = constructInput(xy2[0], xy2[1]);
          var target2 = batch2.y[t2] > 0 ? 1 : -1;
          nn.forwardProp(network, input2);
          clientEndLoss += nn.Errors.SQUARE.error(nn.getOutputNode(network).output, target2);
          clientEndSamples++;
        }
      }
      clientEndLoss /= Math.max(1, clientEndSamples);
      clientLossValues.push(clientEndLoss);

      // Delta vs cluster weights
      var wLocalAfter = nnFlattenWeights();
      var deltaArr = new Float32Array(wLocalAfter.length);
      for (var di = 0; di < wLocalAfter.length; di++) deltaArr[di] = wLocalAfter[di] - w0c[di];
      var delta: Weights = [deltaArr];

      // Client-level DP
      if (cfg.dpClientLevel && cfg.dpClipNorm && cfg.dpClipNorm > 0) {
        delta = clipUpdate(delta, cfg.dpClipNorm);
        var dpNoiseMult = (cfg.dpNoiseMult !== undefined && cfg.dpNoiseMult !== null) ? cfg.dpNoiseMult : 0;
        var sigma = dpNoiseMult * cfg.dpClipNorm;
        if (sigma > 0) {
          delta = addGaussianNoise(delta, sigma);
          flMetrics.dpPrivacyBudget += 1 / (sigma * sigma);
        }
      }

      perCluster[cid].push({delta: delta, weight: client.size, clientLoss: clientEndLoss});
      flLastClientDelta[clientIdx] = delta[0];
    }

    // Aggregate per cluster and update server cluster models
    for (var c = 0; c < K; c++) {
      var deltasC = perCluster[c];
      if (deltasC.length === 0) continue;

      var agg = aggregateDeltas(deltasC.map(function(d){ return {delta: d.delta, weight: d.weight}; }), !!cfg.weightedAggregation);

      // Server-level DP
      if (!cfg.dpClientLevel && cfg.dpClipNorm && cfg.dpClipNorm > 0) {
        agg = clipUpdate(agg, cfg.dpClipNorm);
        var dpNoiseMult2 = (cfg.dpNoiseMult !== undefined && cfg.dpNoiseMult !== null) ? cfg.dpNoiseMult : 0;
        var sigma2 = dpNoiseMult2 * cfg.dpClipNorm;
        if (sigma2 > 0) {
          agg = addGaussianNoise(agg, sigma2);
          flMetrics.dpPrivacyBudget += 1 / (sigma2 * sigma2);
        }
      }

      // FedAvg/FedProx server update: w_c += mean(delta)
      var wc = flClusterWeights![c];
      for (var wi = 0; wi < wc.length; wi++) wc[wi] = wc[wi] + agg[0][wi];
    }

    // Metrics and UI updates
    var roundEndTime = performance.now();
    var roundTime = roundEndTime - roundStartTime;

    var nFloats = flClusterWeights![0].length;
    var commBytes = (nFloats * 4) * (roundClients.length + K);

    flMetrics.participationRates.push([clientParticipationRate, participationRate]);
    flMetrics.commCosts.push([commBytes / 1024]); // KB
    if (clientLossValues.length > 0) {
      flMetrics.clientLosses.push([
        Math.max.apply(null, clientLossValues),
        clientLossValues.reduce(function(a, b){ return a + b; }, 0) / clientLossValues.length,
        Math.min.apply(null, clientLossValues)
      ]);
    }
    if (iter > 0) {
      var lineChartData = lineChart.getData();
      if (lineChartData.length > 1) {
        var prevLoss = lineChartData[lineChartData.length - 2].y[0];
        var currentLoss = lossTrain;
        var convergenceRate = Math.max(0, (prevLoss - currentLoss) / Math.max(prevLoss, 1e-12));
        flMetrics.convergenceRates.push([convergenceRate]);
      }
    }
    updateFLCharts();

    // Update quick UI stats
    var fmt = function(n:number) {
      if (n < 1024) return (n|0) + " B";
      if (n < 1024*1024) return (n/1024).toFixed(1) + " KB";
      if (n < 1024*1024*1024) return (n/1024/1024).toFixed(1) + " MB";
      return (n/1024/1024/1024).toFixed(2) + " GB";
    };
    var setTxt = function(id:string, v:string){ var el = document.getElementById(id); if (el) el.textContent = v; };
    setTxt("fl-round", String(iter + 1));
    setTxt("fl-participating", String(roundClients.length));
    setTxt("fl-comm", fmt(commBytes));
    setTxt("fl-participation-rate", (participationRate * 100).toFixed(1) + "%");
    setTxt("fl-client-fairness", clientLossValues.length > 0 ? 
      ("σ=" + (Math.sqrt(clientLossValues.reduce(function(acc, val) {
        var mean = clientLossValues.reduce(function(a, b){ return a + b; }, 0) / clientLossValues.length;
        return acc + (val - mean) * (val - mean);
      }, 0) / clientLossValues.length)).toFixed(3)) : "N/A");

    // Visualization: use cluster-size-weighted average for display
    var viewFlat = averageClusterWeightsBySize();
    nnSetWeightsFromFlat(viewFlat);

    // Eval
    lossTrain = getLoss(network, trainData);
    lossTest  = getLoss(network, testData);
    iter++;
    flRoundsSinceCluster++;
    reclusterIfNeeded(cfg);
    updateUI();
    return;
  }

  // -------- Single-model path (FedAvg, FedProx, SCAFFOLD, FedAdam) --------

  // Global weights at round start
  var w0 = nnCloneWeights()[0];
  var roundStartTime = performance.now();

  // SCAFFOLD: initialize control variates if needed
  if (cfg.algo === "SCAFFOLD") {
    if (!flScaffoldC || flScaffoldC.length !== w0.length) flScaffoldC = new Float32Array(w0.length);
    if (flScaffoldCi.length !== flClients.length) {
      flScaffoldCi = new Array(flClients.length);
      for (var i0 = 0; i0 < flClients.length; i0++) flScaffoldCi[i0] = new Float32Array(w0.length);
    } else {
      for (var i1 = 0; i1 < flClients.length; i1++) {
        if (!flScaffoldCi[i1] || flScaffoldCi[i1]!.length !== w0.length) flScaffoldCi[i1] = new Float32Array(w0.length);
      }
    }
  }

  // Sample clients
  var k = Math.max(1, Math.round(cfg.clientFrac * flClients.length));
  var shuffled = flClients.slice(0);
  shuffled.sort(function(){ return Math.random() - 0.5; });
  var dropout = (cfg.clientDropout !== undefined && cfg.clientDropout !== null) ? cfg.clientDropout : 0;
  var roundClients: ClientData[] = [];
  var selectedClients = 0;
  for (var i = 0; i < k && i < shuffled.length; i++) {
    selectedClients++;
    if (Math.random() > dropout) roundClients.push(shuffled[i]);
  }
  if (roundClients.length === 0 && shuffled.length > 0) roundClients.push(shuffled[0]);

  var participationRate = roundClients.length / selectedClients;
  var clientParticipationRate = selectedClients / flClients.length;

  var clientLossValues: number[] = [];
  var deltas: {delta: Weights; weight: number; clientLoss: number}[] = [];
  var deltaCs: {delta: Weights; weight: number}[] = []; // For SCAFFOLD

  // Each client trains locally
  for (var rc = 0; rc < roundClients.length; rc++) {
    var client = roundClients[rc];
    var clientStartLoss = 0;
    var clientSamples = 0;

    // Set local weights to w0
    var wLocalFlat = new Float32Array(w0.length);
    for (var ii = 0; ii < w0.length; ii++) wLocalFlat[ii] = w0[ii];
    nnSetWeightsFromFlat(wLocalFlat);

    // Initial client loss
    for (var b = 0; b < client.batches.length; b++) {
      var batch = client.batches[b];
      for (var t = 0; t < batch.x.length; t++) {
        var xy = batch.x[t];
        var input = constructInput(xy[0], xy[1]);
        var target = batch.y[t] > 0 ? 1 : -1;
        nn.forwardProp(network, input);
        clientStartLoss += nn.Errors.SQUARE.error(nn.getOutputNode(network).output, target);
        clientSamples++;
      }
    }
    clientStartLoss /= clientSamples;

    // Local training
    for (var e = 0; e < cfg.localEpochs; e++) {
      for (var b = 0; b < client.batches.length; b++) {
        var batch = client.batches[b];
        for (var t = 0; t < batch.x.length; t++) {
          var xy = batch.x[t];
          var input = constructInput(xy[0], xy[1]);
          var target = batch.y[t] > 0 ? 1 : -1;
          nn.forwardProp(network, input);
          nn.backProp(network, target, nn.Errors.SQUARE);
        }

        // FedProx proximal term
        if (cfg.algo === "FedProx" && cfg.mu && cfg.mu > 0) {
          var idx = 0;
          for (var layerIdx = 1; layerIdx < network.length; layerIdx++) {
            var layer = network[layerIdx];
            for (var i3 = 0; i3 < layer.length; i3++) {
              var node = layer[i3];
              var gBias = cfg.mu * (node.bias - w0[idx++]);
              node.accInputDer += gBias;
              node.numAccumulatedDers += 1;
            }
            for (var i4 = 0; i4 < layer.length; i4++) {
              var node2 = layer[i4];
              for (var j = 0; j < node2.inputLinks.length; j++) {
                var link = node2.inputLinks[j];
                var gW = cfg.mu * (link.weight - w0[idx++]);
                link.accErrorDer += gW;
                link.numAccumulatedDers += 1;
              }
            }
          }
        }

        // SCAFFOLD correction
        if (cfg.algo === "SCAFFOLD" && flScaffoldC && flScaffoldCi.length === flClients.length) {
          var ciIndex = flClients.indexOf(client);
          var ciVec = (ciIndex >= 0 && flScaffoldCi[ciIndex]) ? flScaffoldCi[ciIndex]! : new Float32Array(w0.length);
          var idx2 = 0;
          for (var layerIdx2 = 1; layerIdx2 < network.length; layerIdx2++) {
            var layer2 = network[layerIdx2];
            for (var nb = 0; nb < layer2.length; nb++) {
              var nodeB = layer2[nb];
              var gB = flScaffoldC[idx2] - ciVec[idx2];
              nodeB.accInputDer += gB;
              nodeB.numAccumulatedDers += 1;
              idx2++;
            }
            for (var nw = 0; nw < layer2.length; nw++) {
              var nodeW = layer2[nw];
              for (var jw = 0; jw < nodeW.inputLinks.length; jw++) {
                var linkW = nodeW.inputLinks[jw];
                var gW2 = flScaffoldC[idx2] - ciVec[idx2];
                linkW.accErrorDer += gW2;
                linkW.numAccumulatedDers += 1;
                idx2++;
              }
            }
          }
        }

        nn.updateWeights(network, cfg.clientLR, state.regularizationRate);
      }
    }

    // Final client loss
    var clientEndLoss = 0;
    var clientEndSamples = 0;
    for (var bEnd = 0; bEnd < client.batches.length; bEnd++) {
      var batchEnd = client.batches[bEnd];
      for (var tEnd = 0; tEnd < batchEnd.x.length; tEnd++) {
        var xyEnd = batchEnd.x[tEnd];
        var inputEnd = constructInput(xyEnd[0], xyEnd[1]);
        var targetEnd = batchEnd.y[tEnd] > 0 ? 1 : -1;
        nn.forwardProp(network, inputEnd);
        clientEndLoss += nn.Errors.SQUARE.error(nn.getOutputNode(network).output, targetEnd);
        clientEndSamples++;
      }
    }
    clientEndLoss /= clientEndSamples;
    clientLossValues.push(clientEndLoss);

    // Delta calculation
    var wLocalAfter = nnFlattenWeights();
    var deltaArr = new Float32Array(wLocalAfter.length);
    for (var di = 0; di < wLocalAfter.length; di++) deltaArr[di] = wLocalAfter[di] - w0[di];
    var delta: Weights = [deltaArr];

    // Client-level DP
    if (cfg.dpClientLevel && cfg.dpClipNorm && cfg.dpClipNorm > 0) {
      delta = clipUpdate(delta, cfg.dpClipNorm);
      var dpNoiseMult = (cfg.dpNoiseMult !== undefined && cfg.dpNoiseMult !== null) ? cfg.dpNoiseMult : 0;
      var sigma = dpNoiseMult * cfg.dpClipNorm;
      if (sigma > 0) {
        delta = addGaussianNoise(delta, sigma);
        flMetrics.dpPrivacyBudget += 1 / (sigma * sigma);
      }
    }

    deltas.push({delta: delta, weight: client.size, clientLoss: clientEndLoss});

    // SCAFFOLD: compute Δc_i and update local c_i
    if (cfg.algo === "SCAFFOLD" && flScaffoldC && flScaffoldCi.length === flClients.length) {
      var ciIdx = flClients.indexOf(client);
      if (ciIdx >= 0) {
        var denom = Math.max(1, cfg.localEpochs) * Math.max(1e-8, cfg.clientLR);
        var deltaCiArr = new Float32Array(w0.length);
        for (var k2 = 0; k2 < w0.length; k2++) {
          deltaCiArr[k2] = -flScaffoldC[k2] + (w0[k2] - wLocalAfter[k2]) / denom;
        }
        var ciVec2 = flScaffoldCi[ciIdx]!;
        for (var k3 = 0; k3 < ciVec2.length; k3++) ciVec2[k3] += deltaCiArr[k3];
        flScaffoldCi[ciIdx] = ciVec2;
        deltaCs.push({delta: [deltaCiArr], weight: client.size});
      }
    }
  }

  // Aggregate and update server
  var agg = aggregateDeltas(deltas.map(function(d){ return {delta: d.delta, weight: d.weight}; }), !!cfg.weightedAggregation);

  // Server-level DP
  if (!cfg.dpClientLevel && cfg.dpClipNorm && cfg.dpClipNorm > 0) {
    agg = clipUpdate(agg, cfg.dpClipNorm);
    var dpNoiseMult2 = (cfg.dpNoiseMult !== undefined && cfg.dpNoiseMult !== null) ? cfg.dpNoiseMult : 0;
    var sigma2 = dpNoiseMult2 * cfg.dpClipNorm;
    if (sigma2 > 0) {
      agg = addGaussianNoise(agg, sigma2);
      flMetrics.dpPrivacyBudget += 1 / (sigma2 * sigma2);
    }
  }

  // Server update
  if (cfg.algo === "FedAdam") {
    if (!flServerAdam) {
      flServerAdam = new ServerAdam(
        (cfg.serverLR !== undefined && cfg.serverLR !== null) ? cfg.serverLR : 0.01,
        (cfg.beta1    !== undefined && cfg.beta1    !== null) ? cfg.beta1    : 0.9,
        (cfg.beta2    !== undefined && cfg.beta2    !== null) ? cfg.beta2    : 0.999,
        (cfg.eps      !== undefined && cfg.eps      !== null) ? cfg.eps      : 1e-8
      );
    }
    var grad = [new Float32Array(agg[0].length)];
    for (var gi = 0; gi < agg[0].length; gi++) grad[0][gi] = -agg[0][gi];
    var newW = flServerAdam.step([w0], grad);
    nnSetWeights(newW);
  } else {
    // FedAvg, FedProx, SCAFFOLD: w <- w + mean(delta)
    var newFlat = new Float32Array(w0.length);
    for (var iNew = 0; iNew < w0.length; iNew++) newFlat[iNew] = w0[iNew] + agg[0][iNew];
    nnSetWeightsFromFlat(newFlat);

    // SCAFFOLD: update server control variate c
    if (cfg.algo === "SCAFFOLD" && deltaCs.length > 0 && flScaffoldC) {
      var aggC = aggregateDeltas(deltaCs, !!cfg.weightedAggregation);
      for (var i4 = 0; i4 < flScaffoldC.length; i4++) flScaffoldC[i4] += aggC[0][i4];
    }
  }

  // Metrics
  var roundEndTime = performance.now();
  var roundTime = roundEndTime - roundStartTime;
  var nFloats = w0.length;
  var commBytes = (nFloats * 4) * (roundClients.length + 1);

  flMetrics.participationRates.push([clientParticipationRate, participationRate]);
  flMetrics.commCosts.push([commBytes / 1024]);
  if (clientLossValues.length > 0) {
    flMetrics.clientLosses.push([
      Math.max.apply(null, clientLossValues),
      clientLossValues.reduce(function(a, b){ return a + b; }, 0) / clientLossValues.length,
      Math.min.apply(null, clientLossValues)
    ]);
  }
  if (iter > 0) {
    var lineChartData = lineChart.getData();
    if (lineChartData.length > 1) {
      var prevLoss = lineChartData[lineChartData.length - 2].y[0];
      var currentLoss = lossTrain;
      var convergenceRate = Math.max(0, (prevLoss - currentLoss) / prevLoss);
      flMetrics.convergenceRates.push([convergenceRate]);
    }
  }
  updateFLCharts();

  // UI quick stats
  var fmt = function(n:number) {
    if (n < 1024) return (n|0) + " B";
    if (n < 1024*1024) return (n/1024).toFixed(1) + " KB";
    if (n < 1024*1024*1024) return (n/1024/1024).toFixed(1) + " MB";
    return (n/1024/1024/1024).toFixed(2) + " GB";
  };
  var set = function(id:string, v:string){ var el = document.getElementById(id); if (el) el.textContent = v; };
  set("fl-round", String(iter + 1));
  set("fl-participating", String(roundClients.length));
  set("fl-comm", fmt(commBytes));
  set("fl-participation-rate", (participationRate * 100).toFixed(1) + "%");
  set("fl-client-fairness", clientLossValues.length > 0 ? 
    ("σ=" + (Math.sqrt(clientLossValues.reduce(function(acc, val) {
      var mean = clientLossValues.reduce(function(a, b){ return a + b; }, 0) / clientLossValues.length;
      return acc + (val - mean) * (val - mean);
    }, 0) / clientLossValues.length)).toFixed(3)) : "N/A");

  // Eval & UI
  lossTrain = getLoss(network, trainData);
  lossTest  = getLoss(network, testData);
  iter++;
  updateUI();
}

function oneStepSGD(): void {
  iter++;
  trainData.forEach((point, i) => {
    let input = constructInput(point.x, point.y);
    nn.forwardProp(network, input);
    nn.backProp(network, point.label, nn.Errors.SQUARE);
    if ((i + 1) % state.batchSize === 0) {
      nn.updateWeights(network, state.learningRate, state.regularizationRate);
    }
  });
  // Compute the loss.
  lossTrain = getLoss(network, trainData);
  lossTest = getLoss(network, testData);
  updateUI();
}

export function getOutputWeights(network: nn.Node[][]): number[] {
  let weights: number[] = [];
  for (let layerIdx = 0; layerIdx < network.length - 1; layerIdx++) {
    let currentLayer = network[layerIdx];
    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i];
      for (let j = 0; j < node.outputs.length; j++) {
        let output = node.outputs[j];
        weights.push(output.weight);
      }
    }
  }
  return weights;
}
function reset(onStartup=false) {
  lineChart.reset();
  
  // Reset FL charts - FIXED
  if (isFLEnabled()) {
    // Use for...in loop instead of Object.values()
    for (let chartName in flCharts) {
      if (flCharts.hasOwnProperty(chartName)) {
        flCharts[chartName].reset();
      }
    }
    flMetrics = {
      participationRates: [],
      commCosts: [],
      clientLosses: [],
      convergenceRates: [],
      fairnessMetrics: [],
      dpPrivacyBudget: 0
    };

    flClusterWeights = null;
    flClientCluster = [];
    flLastClientDelta = [];
    flClusterCount = 1;
    flRoundsSinceCluster = 0;
  }
  
  state.serialize();
  if (!onStartup) {
    userHasInteracted();
  }
  player.pause();

  flClients = null;
  flServerAdam = null;
  flLastSig = "";

  let suffix = state.numHiddenLayers !== 1 ? "s" : "";
  d3.select("#layers-label").text("Hidden layer" + suffix);
  d3.select("#num-layers").text(state.numHiddenLayers);

  // Make a simple network.
  iter = 0;
  let numInputs = constructInput(0 , 0).length;
  let shape = [numInputs].concat(state.networkShape).concat([1]);
  let outputActivation = (state.problem === Problem.REGRESSION) ?
      nn.Activations.LINEAR : nn.Activations.TANH;
  network = nn.buildNetwork(shape, state.activation, outputActivation,
      state.regularization, constructInputIds(), state.initZero);
  lossTrain = getLoss(network, trainData);
  lossTest = getLoss(network, testData);
  drawNetwork(network);
  updateUI(true);
}

function initTutorial() {
  if (state.tutorial == null || state.tutorial === '' || state.hideText) {
    return;
  }
  // Remove all other text.
  d3.selectAll("article div.l--body").remove();
  let tutorial = d3.select("article").append("div")
    .attr("class", "l--body");
  // Insert tutorial text.
  d3.html(`tutorials/${state.tutorial}.html`, (err, htmlFragment) => {
    if (err) {
      throw err;
    }
    tutorial.node().appendChild(htmlFragment);
    // If the tutorial has a <title> tag, set the page title to that.
    let title = tutorial.select("title");
    if (title.size()) {
      d3.select("header h1").style({
        "margin-top": "20px",
        "margin-bottom": "20px",
      })
      .text(title.text());
      document.title = title.text();
    }
  });
}

function drawDatasetThumbnails() {
  function renderThumbnail(canvas, dataGenerator) {
    let w = 100;
    let h = 100;
    canvas.setAttribute("width", w);
    canvas.setAttribute("height", h);
    let context = canvas.getContext("2d");
    let data = dataGenerator(200, 0);
    data.forEach(function(d) {
      context.fillStyle = colorScale(d.label);
      context.fillRect(w * (d.x + 6) / 12, h * (d.y + 6) / 12, 4, 4);
    });
    d3.select(canvas.parentNode).style("display", null);
  }
  d3.selectAll(".dataset").style("display", "none");

  if (state.problem === Problem.CLASSIFICATION) {
    for (let dataset in datasets) {
      let canvas: any =
          document.querySelector(`canvas[data-dataset=${dataset}]`);
      let dataGenerator = datasets[dataset];
      renderThumbnail(canvas, dataGenerator);
    }
  }
  if (state.problem === Problem.REGRESSION) {
    for (let regDataset in regDatasets) {
      let canvas: any =
          document.querySelector(`canvas[data-regDataset=${regDataset}]`);
      let dataGenerator = regDatasets[regDataset];
      renderThumbnail(canvas, dataGenerator);
    }
  }
}

function hideControls() {
  // Set display:none to all the UI elements that are hidden.
  let hiddenProps = state.getHiddenProps();
  hiddenProps.forEach(prop => {
    let controls = d3.selectAll(`.ui-${prop}`);
    if (controls.size() === 0) {
      console.warn(`0 html elements found with class .ui-${prop}`);
    }
    controls.style("display", "none");
  });

  // Also add checkbox for each hidable control in the "use it in classrom"
  // section.
  let hideControls = d3.select(".hide-controls");
  HIDABLE_CONTROLS.forEach(([text, id]) => {
    let label = hideControls.append("label")
      .attr("class", "mdl-checkbox mdl-js-checkbox mdl-js-ripple-effect");
    let input = label.append("input")
      .attr({
        type: "checkbox",
        class: "mdl-checkbox__input",
      });
    if (hiddenProps.indexOf(id) === -1) {
      input.attr("checked", "true");
    }
    input.on("change", function() {
      state.setHideProperty(id, !this.checked);
      state.serialize();
      userHasInteracted();
      d3.select(".hide-controls-link")
        .attr("href", window.location.href);
    });
    label.append("span")
      .attr("class", "mdl-checkbox__label label")
      .text(text);
  });
  d3.select(".hide-controls-link")
    .attr("href", window.location.href);
}

function generateData(firstTime = false) {
  if (!firstTime) {
    // Change the seed.
    state.seed = Math.random().toFixed(5);
    state.serialize();
    userHasInteracted();
  }
  Math.seedrandom(state.seed);
  let numSamples = (state.problem === Problem.REGRESSION) ?
      NUM_SAMPLES_REGRESS : NUM_SAMPLES_CLASSIFY;
  let generator = state.problem === Problem.CLASSIFICATION ?
      state.dataset : state.regDataset;
  let data = generator(numSamples, state.noise / 100);
  // Shuffle the data in-place.
  shuffle(data);
  // Split into train and test data.
  let splitIndex = Math.floor(data.length * state.percTrainData / 100);
  trainData = data.slice(0, splitIndex);
  testData = data.slice(splitIndex);
  heatMap.updatePoints(trainData);
  heatMap.updateTestPoints(state.showTestData ? testData : []);

  flClients = null;
}

let firstInteraction = true;
let parametersChanged = false;

function userHasInteracted() {
  if (!firstInteraction) {
    return;
  }
  firstInteraction = false;
  let page = 'index';
  if (state.tutorial != null && state.tutorial !== '') {
    page = `/v/tutorials/${state.tutorial}`;
  }
  ga('set', 'page', page);
  ga('send', 'pageview', {'sessionControl': 'start'});
}

function simulationStarted() {
  ga('send', {
    hitType: 'event',
    eventCategory: 'Starting Simulation',
    eventAction: parametersChanged ? 'changed' : 'unchanged',
    eventLabel: state.tutorial == null ? '' : state.tutorial
  });
  parametersChanged = false;
}

drawDatasetThumbnails();
initTutorial();
makeGUI();
generateData(true);
reset(true);
hideControls();
