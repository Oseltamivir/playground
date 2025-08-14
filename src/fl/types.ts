// Minimal, model-agnostic FL contracts you’ll wire to Playground’s model.
export type Weights = Float32Array[]; // one tensor per layer param (flattened)

export interface ClientBatch {
  x: number[][];        // features (2D)
  y: number[];          // labels (0..K-1)
}

export interface ClientData {
  id: number;
  batches: ClientBatch[];  // already batched for faster local loops
  size: number;            // #examples
  classHist?: number[];    // optional: for UI
}

export type Algo = "FedAvg" | "FedAdam" | "FedProx" | "SCAFFOLD";
export type SimilarityMetric = "cosine" | "l2";


export interface FLConfig {
  algo: Algo;
  numClients: number;          // N
  clientFrac: number;          // C in (0,1]
  localEpochs: number;         // E
  batchSize: number;
  clientLR: number;
  clientOptimizer: "sgd" | "momentum"; // simple, extend if needed
  weightedAggregation: boolean; // weight by |D_k|
  clientDropout: number;        // prob a sampled client silently drops
  iidAlpha: number;             // Dirichlet α (∞ ~ IID, small ~ highly non-IID)
  
  // FedProx
  mu?: number;                  // proximal strength

  // FedAdam (server-side)
  serverLR?: number;
  beta1?: number;
  beta2?: number;
  eps?: number;

  // DP (client- or server-level clipping + Gaussian noise)
  dpClipNorm?: number;          // L2 clip on client update
  dpNoiseMult?: number;         // σ; effective noise = σ * clip
  dpClientLevel?: boolean;      // if false, apply at server on aggregated update

  // Clustering
  clusteringEnabled?: boolean;
  numClusters?: number;        // fixed K (>=1); 1 == disabled behavior
  reclusterEvery?: number;     // rounds between reclustering
  warmupRounds?: number;       // rounds before first recluster
  similarityMetric?: SimilarityMetric; // "cosine" | "l2"

}

export interface RoundMetrics {
  round: number;
  participating: number;
  globalAcc?: number;
  globalLoss?: number;
  perClientAcc?: number[]; // optional
  commBytes?: number;
}

export interface FLCallbacks {
  // Required glue to the existing NN:
  cloneWeights(): Weights;
  setWeights(w: Weights): void;
  // Train 1 local epoch over one client with starting weights w0.
  // If algo==="FedProx", apply proximal term grad += mu*(w - w0).
  localTrainOneEpoch(w0: Weights, client: ClientData, cfg: FLConfig): Weights;
  // Eval the global model on the held-out eval split in Playground
  evalGlobal(): {acc: number; loss: number};

  // UI hooks (all optional)
  onRoundEnd?(m: RoundMetrics, w: Weights): void;
  onClientHistograms?(hists: number[][]): void;
}
