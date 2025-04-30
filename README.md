# Inductive Node Classification (GraphSAGE)

| Date       | Day | Task                                                                            |
| ---------- | --- | ------------------------------------------------------------------------------- |
| 2025-05-01 | 1   | Project kickoff: define objectives, set up repo, install dependencies           |
| 2025-05-02 | 2   | Environment validation: CPU profiling tools, basic PyTorch & PyG test scripts   |
| 2025-05-03 | 3   | Download OGBN-Products dataset, inspect raw files                               |
| 2025-05-04 | 4   | Load data in PyG, explore graph stats (nodes, edges, features)                  |
| 2025-05-05 | 5   | Preprocessing script: filter/clean data, basic train/val/test split             |
| 2025-05-06 | 6   | Implement CSR adjacency conversion & save to disk                               |
| 2025-05-07 | 7   | Write data loader using NumPy memmaps & SciPy sparse                            |
| 2025-05-08 | 8   | Integrate PyG NeighborSampler; test small-batch sampling                        |
| 2025-05-09 | 9   | Build baseline 2-layer GraphSAGE model skeleton                                 |
| 2025-05-10 | 10  | Forward pass test on tiny subset (1k nodes)                                     |
| 2025-05-11 | 11  | Implement training loop stub: loss, backward, optimizer                         |
| 2025-05-12 | 12  | CPU multi-threading: tune `torch.set_num_threads` & DataLoader `num_workers`    |
| 2025-05-13 | 13  | End-to-end tiny-run: train 1 epoch on 10k-node subgraph                         |
| 2025-05-14 | 14  | Validation pipeline: compute accuracy on held-out nodes                         |
| 2025-05-15 | 15  | Logging & checkpointing: integrate TensorBoard or simple CSV logging            |
| 2025-05-16 | 16  | Hyperparameter grid: define ranges for lr, hidden dims, sampling sizes          |
| 2025-05-17 | 17  | Automate HP search loop for small grid on 10k subgraph                          |
| 2025-05-18 | 18  | Analyze HP results: select best config for mid-scale run                        |
| 2025-05-19 | 19  | Scale up to 50k-node subgraph; adjust batch\_size & sampling sizes              |
| 2025-05-20 | 20  | Train 5 epochs on 50k; profile CPU & RAM usage                                  |
| 2025-05-21 | 21  | Implement gradient accumulation for larger effective batch                      |
| 2025-05-22 | 22  | Add dropout & L2 regularization; re-evaluate on 50k subgraph                    |
| 2025-05-23 | 23  | Experiment: compare sampling sizes [15,5] vs [10,5]                             |
| 2025-05-24 | 24  | Integrate mixed-precision CPU with `torch.autocast` if supported                |
| 2025-05-25 | 25  | Write data caching layer: HDF5-based feature storage                            |
| 2025-05-26 | 26  | Refactor code into clear modules: data/, models/, train/                        |
| 2025-05-27 | 27  | Implement early stopping based on validation loss                               |
| 2025-05-28 | 28  | Midpoint review: document findings, update README with current benchmarks       |
| 2025-05-29 | 29  | Extend GraphSAGE: add mean, max, and LSTM aggregators                           |
| 2025-05-30 | 30  | Benchmark each aggregator type on 20k subgraph                                  |
| 2025-05-31 | 31  | Summarize aggregator results; choose best-performers for full-scale             |
| 2025-06-01 | 32  | Begin full-scale: configure for 100k-node subgraph                              |
| 2025-06-02 | 33  | Train 3 epochs on 100k; record time/epoch                                       |
| 2025-06-03 | 34  | Profile CPU hotspots with `cProfile` & `memory_profiler`                        |
| 2025-06-04 | 35  | Optimize sparse-dense matmuls: tune MKL threads & data layout                   |
| 2025-06-05 | 36  | Add GraphSAGE with attention aggregator (GAT-style)                             |
| 2025-06-06 | 37  | Implement and test attention heads on 20k subgraph                              |
| 2025-06-07 | 38  | Compare attention vs. best-performer aggregator                                 |
| 2025-06-08 | 39  | Integrate layer normalization & residual connections                            |
| 2025-06-09 | 40  | Re-run full-scale 5-epoch training; evaluate impact of normalization            |
| 2025-06-10 | 41  | Introduce graph-level regularization (e.g., virtual nodes)                      |
| 2025-06-11 | 42  | Implement Cluster-GCN sampling; benchmark vs NeighborSampler                    |
| 2025-06-12 | 43  | Add GraphSAINT sampling; test small induced subgraphs                           |
| 2025-06-13 | 44  | Compare all sampling strategies on 50k subgraph                                 |
| 2025-06-14 | 45  | Mid-project report: document sampling comparisons & chosen strategy             |
| 2025-06-15 | 46  | Extend model: 3-layer GraphSAGE with best aggregator & sampling                 |
| 2025-06-16 | 47  | Hyper-tune 3-layer config on 50k subgraph                                       |
| 2025-06-17 | 48  | Add learning rate scheduler (e.g., CosineAnnealingLR)                           |
| 2025-06-18 | 49  | Integrate gradient clipping & advanced optimizers (e.g., AdamW)                 |
| 2025-06-19 | 50  | Implement distributed DataParallel emulation for multi-process CPU              |
| 2025-06-20 | 51  | Performance test: multi-process vs single-process CPU                           |
| 2025-06-21 | 52  | Add automated profiling in CI using GitHub Actions or Colab notebook            |
| 2025-06-22 | 53  | Begin feature engineering: incorporate node degree, clustering coeff.           |
| 2025-06-23 | 54  | Test engineered features on validation split                                    |
| 2025-06-24 | 55  | Integrate advanced loss functions (e.g., focal loss)                            |
| 2025-06-25 | 56  | Compare focal vs cross-entropy on class-imbalanced labels                       |
| 2025-06-26 | 57  | Add uncertainty estimation via Monte Carlo Dropout                              |
| 2025-06-27 | 58  | Evaluate calibration & confidence intervals on validation set                   |
| 2025-06-28 | 59  | Implement model ensembling: average 3 best checkpoints                          |
| 2025-06-29 | 60  | Benchmark ensemble vs single model accuracy & inference latency                 |
| 2025-06-30 | 61  | Extended evaluation: MRR, Hits\@k metrics                                       |
| 2025-07-01 | 62  | Scalability test: push subgraph size to 200k nodes                              |
| 2025-07-02 | 63  | Profile memory & optimize data sharding for 200k                                |
| 2025-07-03 | 64  | Add custom PyTorch op for edge sampling in C++ via PyBind11                     |
| 2025-07-04 | 65  | Benchmark C++ sampling op vs Python implementation                              |
| 2025-07-05 | 66  | Integrate mixed C++/Python neighbor sampler into pipeline                       |
| 2025-07-06 | 67  | Write PyBind11 build scripts & CI integration                                   |
| 2025-07-07 | 68  | Documentation: generate Sphinx docs for data, model, and training modules       |
| 2025-07-08 | 69  | Create tutorial notebook for end-to-end pipeline                                |
| 2025-07-09 | 70  | Add visualization: TSNE/UMAP of learned embeddings                              |
| 2025-07-10 | 71  | Integrate performance dashboard (e.g., Vega-lite) in notebook                   |
| 2025-07-11 | 72  | Deploy lightweight REST inference API using Flask                               |
| 2025-07-12 | 73  | Package model & API into Docker container                                       |
| 2025-07-13 | 74  | Test Docker deployment locally & on remote VM                                   |
| 2025-07-14 | 75  | Write research report: intro, methods, results                                  |
| 2025-07-15 | 76  | Draft paper: abstract, related work, model architecture                         |
| 2025-07-16 | 77  | Incorporate ablation studies: remove features, vary layers                      |
| 2025-07-17 | 78  | Finalize experiment tables & figures                                            |
| 2025-07-18 | 79  | Peer review: share report & code with colleague for feedback                    |
| 2025-07-19 | 80  | Revise code & documentation per feedback                                        |
| 2025-07-20 | 81  | Prepare slides & poster for internal presentation                               |
| 2025-07-21 | 82  | Practice presentation & time run-through                                        |
| 2025-07-22 | 83  | Final code cleanup & repository tagging (v1.0)                                  |
| 2025-07-23 | 84  | Publish repository & docs on GitHub                                             |
| 2025-07-24 | 85  | Optional: submit model to OGB leaderboards                                      |
| 2025-07-25 | 86  | Workshop: record screencast of pipeline for YouTube or internal knowledge share |
| 2025-07-26 | 87  | Plan next steps: new GNN ideas or deployment improvements                       |
| 2025-07-27 | 88  | Retrospective: document lessons learned and future improvements                 |
| 2025-07-28 | 89  | Final validation: rerun full pipeline end-to-end and confirm reproducibility    |
| 2025-07-29 | 90  | Project wrap-up: archive data, finalize report, celebrate completion!           |

---


