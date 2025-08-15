# Federated Learning Playground
This is a fork of [Deep Playground](https://github.com/tensorflow/playground) by TensorFlow.

The changes made include the addition of federated learning (FL) capabilities, allowing users to experiment with FL concepts and algorithms directly within the playground environment. 

## Additions
1. Federated Learning Engine: The core addition is a simulation engine for federated learning. The logic is primarily in `playground.ts` within the `oneStepFL` function, which handles the orchestration of client sampling, local training, and server aggregation.

2. FL Algorithms: There are some new FL algorithms, which can be selected from the UI:
    - FedAvg and FedProx
    - FedAdam (using ServerAdam optimizer)
    - SCAFFOLD

3. Data Partitioning: Simulation of data heterogeneity among clients. The data is partitioned among clients using a pseudo-Dirichlet distribution, controlled by the "Non-IID α" setting.

4. Clustered FL: Supports clustering clients into groups that train separate models. Uses k-Means clustering with either cosine similarity or l2 distance.

5. Differential Privacy: Includes options for adding differential privacy to the training process, with functions like addGaussianNoise and clipUpdate from `dp.ts`.

6. New UI and Visualizations: The UI expanded to include controls for FL parameters and new charts to visualize FL-specific metrics like client participation, communication cost, and loss distribution.

All modifications are done on top of TensorFlow's work, if "Enable FL" is not checked, the original TensorFlow functionality remains.

---
# Deep playground

Deep playground is an interactive visualization of neural networks, written in
TypeScript using d3.js. We use GitHub issues for tracking new requests and bugs.
Your feedback is highly appreciated!

**If you'd like to contribute, be sure to review the [contribution guidelines](CONTRIBUTING.md).**

## Development

To run the visualization locally, run:
- `npm i` to install dependencies
- `npm run build` to compile the app and place it in the `dist/` directory
- `npm run serve` to serve from the `dist/` directory and open a page on your browser.

For a fast edit-refresh cycle when developing run `npm run serve-watch`.
This will start an http server and automatically re-compile the TypeScript,
HTML and CSS files whenever they change.

## For owners
To push to production: `git subtree push --prefix dist origin gh-pages`.

This is not an official Google product.
