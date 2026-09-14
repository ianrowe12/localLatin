## Wins per metric, old vs new

| Metric | Dir | old ABTT wins | new ABTT wins | cells that flipped |
|---|---|---|---|---|
| rho_LOO | higher | 6/6 | 5/6 | PhilTa/MaRC A->b |
| DelAUC gap | higher | 3/6 | 4/6 | LaTa/MaRC A->b, PhilTa/IG b->A, PhilTa/MaRC b->A |
| InsAUC gap | higher | 5/6 | 5/6 | - |
| tau_LOO | higher | 6/6 | 5/6 | PhilTa/MaRC A->b |
| AOPC-Suff | higher | 2/6 | 4/6 | PhilTa/IG b->A, mT5-base/MaRC b->A |
| AOPC-Comp | higher | 3/6 | 4/6 | LaTa/MaRC A->b, PhilTa/IG b->A, PhilTa/MaRC b->A |
| DelAUC | lower | 3/6 | 4/6 | LaTa/MaRC A->b, PhilTa/IG b->A, PhilTa/MaRC b->A |
| InsAUC | higher | 2/6 | 4/6 | PhilTa/IG b->A, mT5-base/MaRC b->A |
| DelAUC random floor | higher | 0/6 | 2/6 | LaTa/IG b->A, LaTa/MaRC b->A |
| Suff@25% | higher | 2/6 | 4/6 | PhilTa/IG b->A, mT5-base/MaRC b->A |
| Comp@25% | higher | 2/6 | 4/6 | PhilTa/IG b->A, PhilTa/MaRC b->A |
| MinFrac@0.80 | lower | 1/6 | 2/6 | PhilTa/IG b->A |

## Shuffled-attribution control (criterion 5)

| Metric | old positive cells | new positive cells | old failures | new failures |
|---|---|---|---|---|
| rho_LOO | 12/12 | 12/12 | none | none |
| tau_LOO | 12/12 | 12/12 | none | none |
| DelAUC gap | 12/12 | 12/12 | none | none |
| InsAUC gap | 10/12 | 11/12 | LaTa/MaRC baseline (-0.189); mT5-base/IG baseline (-0.030) | mT5-base/IG baseline (-0.024) |
| AOPC-Suff | 10/12 | 11/12 | LaTa/MaRC baseline (-0.189); mT5-base/IG baseline (-0.030) | mT5-base/IG baseline (-0.024) |
| AOPC-Comp | 12/12 | 12/12 | none | none |

## Every cell

| Metric | Dir | Cell | old base -> ABTT | new base -> ABTT | old | new |
|---|---|---|---|---|:-:|:-:|
| rho_LOO | up | LaTa/IG | 0.013 -> 0.298 | 0.003 -> 0.370 | A | A |
| rho_LOO | up | LaTa/MaRC | 0.070 -> 0.397 | 0.083 -> 0.538 | A | A |
| rho_LOO | up | PhilTa/IG | 0.144 -> 0.577 | 0.329 -> 0.614 | A | A |
| rho_LOO | up | PhilTa/MaRC | 0.179 -> 0.367 | 0.469 -> 0.445 | A | b **flip** |
| rho_LOO | up | mT5-base/IG | 0.138 -> 0.626 | 0.143 -> 0.686 | A | A |
| rho_LOO | up | mT5-base/MaRC | 0.272 -> 0.415 | 0.257 -> 0.504 | A | A |
| DelAUC gap | up | LaTa/IG | 0.504 -> 0.178 | 0.842 -> 0.180 | b | b |
| DelAUC gap | up | LaTa/MaRC | 0.244 -> 0.287 | 0.506 -> 0.327 | A | b **flip** |
| DelAUC gap | up | PhilTa/IG | 0.806 -> 0.420 | 0.112 -> 0.400 | b | A **flip** |
| DelAUC gap | up | PhilTa/MaRC | 0.758 -> 0.353 | 0.200 -> 0.310 | b | A **flip** |
| DelAUC gap | up | mT5-base/IG | 0.394 -> 0.548 | 0.061 -> 0.561 | A | A |
| DelAUC gap | up | mT5-base/MaRC | 0.075 -> 0.431 | 0.042 -> 0.464 | A | A |
| InsAUC gap | up | LaTa/IG | 0.153 -> 0.032 | 0.265 -> 0.060 | b | b |
| InsAUC gap | up | LaTa/MaRC | -0.201 -> 0.098 | 0.069 -> 0.137 | A | A |
| InsAUC gap | up | PhilTa/IG | 0.118 -> 0.244 | 0.003 -> 0.242 | A | A |
| InsAUC gap | up | PhilTa/MaRC | 0.104 -> 0.200 | 0.061 -> 0.190 | A | A |
| InsAUC gap | up | mT5-base/IG | -0.028 -> 0.353 | -0.024 -> 0.402 | A | A |
| InsAUC gap | up | mT5-base/MaRC | 0.017 -> 0.243 | 0.002 -> 0.329 | A | A |
| tau_LOO | up | LaTa/IG | 0.007 -> 0.212 | 0.007 -> 0.262 | A | A |
| tau_LOO | up | LaTa/MaRC | 0.059 -> 0.292 | 0.068 -> 0.400 | A | A |
| tau_LOO | up | PhilTa/IG | 0.106 -> 0.432 | 0.234 -> 0.462 | A | A |
| tau_LOO | up | PhilTa/MaRC | 0.133 -> 0.258 | 0.334 -> 0.318 | A | b **flip** |
| tau_LOO | up | mT5-base/IG | 0.102 -> 0.465 | 0.101 -> 0.515 | A | A |
| tau_LOO | up | mT5-base/MaRC | 0.186 -> 0.291 | 0.181 -> 0.359 | A | A |
| AOPC-Suff | up | LaTa/IG | 0.972 -> 0.833 | 0.995 -> 0.876 | b | b |
| AOPC-Suff | up | LaTa/MaRC | 0.617 -> 0.898 | 0.799 -> 0.953 | A | A |
| AOPC-Suff | up | PhilTa/IG | 0.979 -> 0.970 | 0.872 -> 0.979 | b | A **flip** |
| AOPC-Suff | up | PhilTa/MaRC | 0.964 -> 0.926 | 0.931 -> 0.927 | b | b |
| AOPC-Suff | up | mT5-base/IG | 0.908 -> 1.040 | 0.944 -> 1.091 | A | A |
| AOPC-Suff | up | mT5-base/MaRC | 0.953 -> 0.930 | 0.970 -> 1.019 | b | A **flip** |
| AOPC-Comp | up | LaTa/IG | 0.718 -> 0.395 | 1.154 -> 0.388 | b | b |
| AOPC-Comp | up | LaTa/MaRC | 0.458 -> 0.504 | 0.818 -> 0.535 | A | b **flip** |
| AOPC-Comp | up | PhilTa/IG | 0.939 -> 0.711 | 0.263 -> 0.684 | b | A **flip** |
| AOPC-Comp | up | PhilTa/MaRC | 0.891 -> 0.643 | 0.351 -> 0.594 | b | A **flip** |
| AOPC-Comp | up | mT5-base/IG | 0.472 -> 0.882 | 0.106 -> 0.876 | A | A |
| AOPC-Comp | up | mT5-base/MaRC | 0.153 -> 0.766 | 0.087 -> 0.778 | A | A |
| DelAUC | down | LaTa/IG | 0.291 -> 0.614 | -0.144 -> 0.621 | b | b |
| DelAUC | down | LaTa/MaRC | 0.550 -> 0.505 | 0.191 -> 0.475 | A | b **flip** |
| DelAUC | down | PhilTa/IG | 0.069 -> 0.297 | 0.746 -> 0.325 | b | A **flip** |
| DelAUC | down | PhilTa/MaRC | 0.117 -> 0.365 | 0.658 -> 0.415 | b | A **flip** |
| DelAUC | down | mT5-base/IG | 0.534 -> 0.124 | 0.900 -> 0.131 | A | A |
| DelAUC | down | mT5-base/MaRC | 0.853 -> 0.241 | 0.919 -> 0.229 | A | A |
| InsAUC | up | LaTa/IG | 0.963 -> 0.824 | 0.986 -> 0.866 | b | b |
| InsAUC | up | LaTa/MaRC | 0.609 -> 0.889 | 0.789 -> 0.943 | A | A |
| InsAUC | up | PhilTa/IG | 0.971 -> 0.962 | 0.864 -> 0.971 | b | A **flip** |
| InsAUC | up | PhilTa/MaRC | 0.956 -> 0.918 | 0.922 -> 0.919 | b | b |
| InsAUC | up | mT5-base/IG | 0.901 -> 1.034 | 0.937 -> 1.085 | A | A |
| InsAUC | up | mT5-base/MaRC | 0.946 -> 0.924 | 0.964 -> 1.012 | b | A **flip** |
| DelAUC random floor | up | LaTa/IG | 0.795 -> 0.792 | 0.697 -> 0.802 | b | A **flip** |
| DelAUC random floor | up | LaTa/MaRC | 0.795 -> 0.792 | 0.697 -> 0.802 | b | A **flip** |
| DelAUC random floor | up | PhilTa/IG | 0.875 -> 0.717 | 0.858 -> 0.725 | b | b |
| DelAUC random floor | up | PhilTa/MaRC | 0.875 -> 0.717 | 0.858 -> 0.725 | b | b |
| DelAUC random floor | up | mT5-base/IG | 0.928 -> 0.671 | 0.961 -> 0.692 | b | b |
| DelAUC random floor | up | mT5-base/MaRC | 0.928 -> 0.671 | 0.961 -> 0.692 | b | b |
| Suff@25% | up | LaTa/IG | 0.965 -> 0.729 | 0.990 -> 0.804 | b | b |
| Suff@25% | up | LaTa/MaRC | 0.556 -> 0.847 | 0.731 -> 0.917 | A | A |
| Suff@25% | up | PhilTa/IG | 0.965 -> 0.941 | 0.814 -> 0.971 | b | A **flip** |
| Suff@25% | up | PhilTa/MaRC | 0.952 -> 0.883 | 0.888 -> 0.878 | b | b |
| Suff@25% | up | mT5-base/IG | 0.885 -> 1.007 | 0.933 -> 1.126 | A | A |
| Suff@25% | up | mT5-base/MaRC | 0.936 -> 0.840 | 0.960 -> 1.013 | b | A **flip** |
| Comp@25% | up | LaTa/IG | 0.658 -> 0.194 | 1.242 -> 0.177 | b | b |
| Comp@25% | up | LaTa/MaRC | 0.329 -> 0.272 | 0.781 -> 0.280 | b | b |
| Comp@25% | up | PhilTa/IG | 0.988 -> 0.546 | 0.113 -> 0.522 | b | A **flip** |
| Comp@25% | up | PhilTa/MaRC | 0.951 -> 0.443 | 0.093 -> 0.363 | b | A **flip** |
| Comp@25% | up | mT5-base/IG | 0.055 -> 0.653 | 0.012 -> 0.642 | A | A |
| Comp@25% | up | mT5-base/MaRC | 0.047 -> 0.591 | 0.014 -> 0.582 | A | A |
| MinFrac@0.80 | down | LaTa/IG | 0.042 -> 0.330 | 0.111 -> 0.272 | b | b |
| MinFrac@0.80 | down | LaTa/MaRC | 0.398 -> 0.291 | 0.184 -> 0.177 | A | A |
| MinFrac@0.80 | down | PhilTa/IG | 0.031 -> 0.144 | 0.240 -> 0.136 | b | A **flip** |
| MinFrac@0.80 | down | PhilTa/MaRC | 0.058 -> 0.296 | 0.149 -> 0.259 | b | b |
| MinFrac@0.80 | down | mT5-base/IG | 0.134 -> 0.232 | 0.086 -> 0.178 | b | b |
| MinFrac@0.80 | down | mT5-base/MaRC | 0.032 -> 0.340 | 0.031 -> 0.246 | b | b |
