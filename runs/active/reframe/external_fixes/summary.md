# External corrections on the full canon testbed

Corrections use only the 439 decretal embeddings, except the two label-free `canon_all`
references, which fit on the same 1,705 witnesses they score. Neither canon embeddings
nor canon labels enter external fits or selection. Coordinate SD uses population SD;
component scaling uses the sample eigenvalues (or their square roots).

Pair AUROC uses all 1,452,660 unordered canon pairs. Acc@1 and MRR exclude singleton
queries, not singleton candidates. Ties in retrieval follow split order (cache metadata
order for decretals). The SD floor count is repeated on both score rows for the same fit.

The raw test-split reproduction gate passed 100/100 layers (maximum absolute AUROC difference 3.06e-07; tolerance 1e-06).

## Canon scores at decretal-selected settings

Each cell shows layer (and D/r where applicable), then AUROC / Acc@1 / MRR.
Each column independently takes the first maximum decretal pair AUROC, with ties
resolved by lowest layer then smallest D/r. `selected.csv` also reports the uncorrected
canon scores at each chosen layer for same-layer comparisons.

| Model | None | Center | ABTT | Coordinate SD | Component SD | Component variance |
|---|---|---|---|---|---|---|
| LaTa | L12: 0.938 / 0.877 / 0.909 | L12: 0.962 / 0.878 / 0.911 | L8/D3: 0.966 / 0.858 / 0.892 | L1: 0.979 / 0.916 / 0.937 | L12/r64: 0.967 / 0.813 / 0.861 | L11/r64: 0.954 / 0.783 / 0.827 |
| PhilTa | L12: 0.899 / 0.828 / 0.866 | L1: 0.980 / 0.935 / 0.952 | L12/D1: 0.974 / 0.906 / 0.929 | L1: 0.983 / 0.950 / 0.962 | L10/r64: 0.944 / 0.694 / 0.759 | L6/r64: 0.964 / 0.778 / 0.827 |
| mT5-base | L12: 0.824 / 0.688 / 0.734 | L12: 0.903 / 0.677 / 0.724 | L6/D3: 0.912 / 0.728 / 0.765 | L11: 0.918 / 0.682 / 0.726 | L6/r64: 0.928 / 0.685 / 0.730 | L6/r64: 0.929 / 0.706 / 0.759 |
| LaBSE | L6: 0.891 / 0.791 / 0.830 | L12: 0.955 / 0.922 / 0.934 | L6/D3: 0.949 / 0.880 / 0.904 | L12: 0.959 / 0.928 / 0.939 | L4/r64: 0.952 / 0.827 / 0.861 | L4/r64: 0.961 / 0.829 / 0.860 |
| Qwen3-0.6B | L19: 0.934 / 0.897 / 0.916 | L21: 0.972 / 0.930 / 0.943 | L25/D1: 0.980 / 0.939 / 0.951 | L19: 0.965 / 0.912 / 0.930 | L16/r64: 0.962 / 0.856 / 0.885 | L5/r64: 0.957 / 0.871 / 0.898 |
| KaLM-mini | L15: 0.906 / 0.845 / 0.878 | L23: 0.973 / 0.941 / 0.951 | L12/D1: 0.942 / 0.841 / 0.877 | L23: 0.977 / 0.942 / 0.953 | L9/r64: 0.963 / 0.846 / 0.878 | L4/r64: 0.959 / 0.859 / 0.885 |

## Restoration and medians across all model-layers

Restored means full-canon AUROC >= 0.90 on a layer whose raw H1
test AUROC was < 0.70.

| Correction | Restored | Median AUROC | Median Acc@1 | Median MRR |
|---|---:|---:|---:|---:|
| `none` | 0/26 | 0.882 | 0.826 | 0.863 |
| `center` | 0/26 | 0.925 | 0.829 | 0.862 |
| `abtt_D1` | 0/26 | 0.950 | 0.859 | 0.888 |
| `abtt_D3` | 23/26 | 0.967 | 0.900 | 0.922 |
| `abtt_D10` | 26/26 | 0.973 | 0.922 | 0.938 |
| `std_coord` | 12/26 | 0.948 | 0.855 | 0.886 |
| `std_comp_r64` | 26/26 | 0.963 | 0.858 | 0.887 |
| `std_comp_r128` | 26/26 | 0.971 | 0.909 | 0.929 |
| `std_comp_r256` | 26/26 | 0.972 | 0.926 | 0.940 |
| `var_comp_r64` | 26/26 | 0.957 | 0.846 | 0.877 |
| `var_comp_r128` | 26/26 | 0.964 | 0.896 | 0.915 |
| `var_comp_r256` | 26/26 | 0.963 | 0.904 | 0.922 |
| `canon_all_abtt_D3` | 26/26 | 0.976 | 0.906 | 0.924 |
| `canon_all_std_coord` | 8/26 | 0.951 | 0.841 | 0.875 |
