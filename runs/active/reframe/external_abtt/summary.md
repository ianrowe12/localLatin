# External ABTT components (decretals)

Leakage: {"n_decr_files": 439, "n_canon_files": 1705, "exact_duplicates": 0, "jaccard5_ge_0.5": 2, "jaccard5_ge_0.3": 3, "jaccard5_median": 0.0755082284607938}

## Subspace overlap with the canon-train components (median over layers)

| model | D | train vs decretals | train vs test | random |
|---|---|---|---|---|
| LaTa | 1 | 0.998 | 0.999 | 0.0013 |
| LaTa | 3 | 0.887 | 0.983 | 0.0039 |
| LaTa | 10 | 0.580 | 0.906 | 0.0130 |
| PhilTa | 1 | 0.933 | 0.991 | 0.0013 |
| PhilTa | 3 | 0.929 | 0.995 | 0.0039 |
| PhilTa | 10 | 0.667 | 0.897 | 0.0130 |
| mT5-base | 1 | 0.999 | 1.000 | 0.0013 |
| mT5-base | 3 | 0.529 | 0.969 | 0.0039 |
| mT5-base | 10 | 0.590 | 0.909 | 0.0130 |
| LaBSE | 1 | 0.488 | 0.994 | 0.0013 |
| LaBSE | 3 | 0.469 | 0.956 | 0.0039 |
| LaBSE | 10 | 0.559 | 0.877 | 0.0130 |
| Qwen3-0.6B | 1 | 0.433 | 0.979 | 0.0010 |
| Qwen3-0.6B | 3 | 0.591 | 0.953 | 0.0029 |
| Qwen3-0.6B | 10 | 0.586 | 0.848 | 0.0098 |
| KaLM-mini | 1 | 0.703 | 0.990 | 0.0011 |
| KaLM-mini | 3 | 0.569 | 0.964 | 0.0033 |
| KaLM-mini | 10 | 0.589 | 0.864 | 0.0112 |

## Collapsed layers (baseline test AUROC < 0.70): 26

| fit | restored >= 0.90 at D=3 | at selected D | median AUROC, selected D |
|---|---|---|---|
| canon_train | 26 | 26 | 0.977 |
| decr | 26 | 26 | 0.965 |
| decr_pcs_train_mean | 19 | 26 | 0.970 |
| canon_all | 26 | 26 | 0.974 |

## All layers, selected D: difference from canon_train

| fit | median dAUROC | min dAUROC | median dDirAcc@1 | min dDirAcc@1 |
|---|---|---|---|---|
| decr | -0.0082 | -0.0296 | -0.0367 | -0.1282 |
| decr_pcs_train_mean | -0.0048 | -0.0752 | -0.0315 | -0.1329 |
| canon_all | -0.0014 | -0.0087 | +0.0064 | -0.0152 |

## Per model, best layer by training DirAcc@1 (selected D)

| model | fit | layer | D | test AUROC | test DirAcc@1 |
|---|---|---|---|---|---|
| LaTa | canon_train | 8 | 10 | 0.965 | 86.1 |
| LaTa | decr | 1 | 10 | 0.978 | 87.4 |
| LaTa | decr_pcs_train_mean | 1 | 10 | 0.975 | 86.7 |
| LaTa | canon_all | 7 | 10 | 0.961 | 87.1 |
| PhilTa | canon_train | 1 | 10 | 0.977 | 88.3 |
| PhilTa | decr | 1 | 10 | 0.979 | 88.2 |
| PhilTa | decr_pcs_train_mean | 1 | 10 | 0.979 | 87.3 |
| PhilTa | canon_all | 1 | 10 | 0.978 | 88.5 |
| mT5-base | canon_train | 1 | 10 | 0.979 | 88.5 |
| mT5-base | decr | 1 | 7 | 0.978 | 87.4 |
| mT5-base | decr_pcs_train_mean | 1 | 10 | 0.978 | 86.7 |
| mT5-base | canon_all | 1 | 10 | 0.976 | 88.0 |
| LaBSE | canon_train | 11 | 10 | 0.987 | 88.5 |
| LaBSE | decr | 11 | 10 | 0.980 | 85.2 |
| LaBSE | decr_pcs_train_mean | 12 | 7 | 0.982 | 86.5 |
| LaBSE | canon_all | 11 | 10 | 0.986 | 87.1 |
| Qwen3-0.6B | canon_train | 5 | 10 | 0.981 | 89.4 |
| Qwen3-0.6B | decr | 1 | 7 | 0.970 | 87.4 |
| Qwen3-0.6B | decr_pcs_train_mean | 24 | 7 | 0.973 | 87.9 |
| Qwen3-0.6B | canon_all | 4 | 10 | 0.984 | 89.3 |
| KaLM-mini | canon_train | 3 | 10 | 0.981 | 89.4 |
| KaLM-mini | decr | 3 | 10 | 0.980 | 88.3 |
| KaLM-mini | decr_pcs_train_mean | 3 | 10 | 0.981 | 88.5 |
| KaLM-mini | canon_all | 1 | 10 | 0.979 | 90.4 |
