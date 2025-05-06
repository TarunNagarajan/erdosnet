### **Erdosnet - Inductive Node Classification**

| Day | Task                                                                            |
| --- | ------------------------------------------------------------------------------- |
| 1   | Project kickoff: define objectives, set up repo, install dependencies           |
| 2   | Environment validation: CPU profiling tools, basic PyTorch & PyG test scripts   |
| 3   | Download OGBN-Products dataset, inspect raw files                               |
| 4   | Load data in PyG, explore graph stats (nodes, edges, features)                  |
| 5   | Preprocessing script: filter/clean data, basic train/val/test split             |
| 6   | Implement CSR adjacency conversion & save to disk                               |
| 7   | Write data loader using NumPy memmaps & SciPy sparse                            |
| 8   | Integrate PyG NeighborSampler; test small-batch sampling                        |
| 9   | Build baseline 2-layer GraphSAGE model skeleton                                 |
| 10  | Forward pass test on tiny subset (1k nodes)                                     |
| 11  | Implement training loop stub: loss, backward, optimizer                         |
| 12  | CPU multi-threading: tune `torch.set_num_threads` & DataLoader `num_workers`    |
| 13  | End-to-end tiny-run: train 1 epoch on 10k-node subgraph                         |
| 14  | Validation pipeline: compute accuracy on held-out nodes                         |
| 15  | Logging & checkpointing: integrate TensorBoard or simple CSV logging            |
| 16  | Hyperparameter grid: define ranges for lr, hidden dims, sampling sizes          |
| 17  | Automate HP search loop for small grid on 10k subgraph                          |
| 18  | Analyze HP results: select best config for mid-scale run                        |
| 19  | Scale up to 50k-node subgraph; adjust batch_size & sampling sizes              |
| 20  | Train 5 epochs on 50k; profile CPU & RAM usage                                  |
| 21  | Implement gradient accumulation for larger effective batch                      |
| 22  | Add dropout & L2 regularization; re-evaluate on 50k subgraph                    |
| 23  | Experiment: compare sampling sizes [15,5] vs [10,5]                             |
| 24  | Integrate mixed-precision CPU with `torch.autocast` if supported                |
| 25  | Write data caching layer: HDF5-based feature storage                            |
| 26  | Refactor code into clear modules: data/, models/, train/                        |
| 27  | Implement early stopping based on validation loss                               |
| 28  | Midpoint review: document findings, update README with current benchmarks       |
| 29  | Extend GraphSAGE: add mean, max, and LSTM aggregators                           |
| 30  | Benchmark each aggregator type on 20k subgraph                                  |
| 31  | Summarize aggregator results; choose best-performers for full-scale             |
| 32  | Begin full-scale: configure for 100k-node subgraph                              |
| 33  | Train 3 epochs on 100k; record time/epoch                                       |
| 34  | Profile CPU hotspots with `cProfile` & `memory_profiler`                        |
| 35  | Optimize sparse-dense matmuls: tune MKL threads & data layout                   |
| 36  | Add GraphSAGE with attention aggregator (GAT-style)                             |
| 37  | Implement and test attention heads on 20k subgraph                              |
| 38  | Compare attention vs. best-performer aggregator                                 |
| 39  | Integrate layer normalization & residual connections                            |
| 40  | Re-run full-scale 5-epoch training; evaluate impact of normalization            |
| 41  | Introduce graph-level regularization (e.g., virtual nodes)                      |
| 42  | Implement Cluster-GCN sampling; benchmark vs NeighborSampler                    |
| 43  | Add GraphSAINT sampling; test small induced subgraphs                           |
| 44  | Compare all sampling strategies on 50k subgraph                                 |
| 45  | Mid-project report: document sampling comparisons & chosen strategy             |
| 46  | Extend model: 3-layer GraphSAGE with best aggregator & sampling                 |
| 47  | Hyper-tune 3-layer config on 50k subgraph                                       |
| 48  | Add learning rate scheduler (e.g., CosineAnnealingLR)                           |
| 49  | Integrate gradient clipping & advanced optimizers (e.g., AdamW)                 |
| 50  | Implement distributed DataParallel emulation for multi-process CPU              |
| 51  | Performance test: multi-process vs single-process CPU                           |
| 52  | Add automated profiling in CI using GitHub Actions or Colab notebook            |
| 53  | Begin feature engineering: incorporate node degree, clustering coeff.           |
| 54  | Test engineered features on validation split                                    |
| 55  | Integrate advanced loss functions (e.g., focal loss)                            |
| 56  | Compare focal vs cross-entropy on class-imbalanced labels                       |
| 57  | Add uncertainty estimation via Monte Carlo Dropout                              |
| 58  | Evaluate calibration & confidence intervals on validation set                   |
| 59  | Implement model ensembling: average 3 best checkpoints                          |
| 60  | Benchmark ensemble vs single model accuracy & inference latency                 |
| 61  | Extended evaluation: MRR, Hits@k metrics                                       |
| 62  | Scalability test: push subgraph size to 200k nodes                              |
| 63  | Profile memory & optimize data sharding for 200k                                |
| 64  | Add custom PyTorch op for edge sampling in C++ via PyBind11                     |
| 65  | Benchmark C++ sampling op vs Python implementation                              |
| 66  | Integrate mixed C++/Python neighbor sampler into pipeline                       |
| 67  | Write PyBind11 build scripts & CI integration                                   |
| 68  | Documentation: generate Sphinx docs for data, model, and training modules       |
| 69  | Create tutorial notebook for end-to-end pipeline                                |
| 70  | Add visualization: TSNE/UMAP of learned embeddings                              |
| 71  | Integrate performance dashboard (e.g., Vega-lite) in notebook                   |
| 72  | Deploy lightweight REST inference API using Flask                               |
| 73  | Package model & API into Docker container                                       |
| 74  | Test Docker deployment locally & on remote VM                                   |
| 75  | Write research report: intro, methods, results                                  |
| 76  | Draft paper: abstract, related work, model architecture                         |
| 77  | Incorporate ablation studies: remove features, vary layers                      |
| 78  | Finalize experiment tables & figures                                            |
| 79  | Peer review: share report & code with colleague for feedback                    |
| 80  | Revise code & documentation per feedback                                        |
| 81  | Prepare slides & poster for internal presentation                               |
| 82  | Practice presentation & time run-through                                        |
| 83  | Final code cleanup & repository tagging (v1.0)                                  |
| 84  | Publish repository & docs on GitHub                                             |
| 85  | Optional: submit model to OGB leaderboards                                      |
| 86  | Workshop: record screencast of pipeline for YouTube or internal knowledge share |
| 87  | Plan next steps: new GNN ideas or deployment improvements                       |
| 88  | Retrospective: document lessons learned and future improvements                 |
| 89  | Final validation: rerun full pipeline end-to-end and confirm reproducibility    |
| 90  | Project wrap-up: archive data, finalize report, celebrate completion!           |
