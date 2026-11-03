# The S2MPJ problem suite

Every S2MPJ problem with an objective, equality constraints only and at most
10 variables, at S2MPJ commit `fea6a70048eaad28b13a08703ddbfdbf65cd9c30`.
12 of the 72 are generated as benchmark experiments here; the
rules in `benchmarks/generate_s2mpj.py` (version 1) refuse the rest,
with the reason. #181 tracks lifting the reasons. Don't edit this file: regenerate it
with `pixi run -e benchmarks generate-s2mpj`.

| problem | n | m | outcome | detail |
|---|---|---|---|---|
| ALLINITC | 4 | 1 | refused: no numeric SOLTN | none stated |
| ALSOTAME | 2 | 1 | refused: only linear constraints | no element sum to hide |
| BT1 | 2 | 1 | refused: hidden constraint is its hidden term plus a constant | CON1 leaves g a function of y alone |
| BT10 | 2 | 2 | refused: no degrees of freedom | n = 2, m = 2 |
| BT11 | 5 | 3 | generated: [bt11.py](bt11.py) | hides CON1 |
| BT12 | 5 | 3 | generated: [bt12.py](bt12.py) | hides CON1 |
| BT13 | 5 | 1 | refused: hidden constraint is its hidden term plus a constant | CON1 leaves g a function of y alone |
| BT2 | 3 | 1 | refused: hidden constraint is its hidden term plus a constant | CON1 leaves g a function of y alone |
| BT3 | 5 | 3 | refused: only linear constraints | no element sum to hide |
| BT4 | 3 | 2 | refused: no numeric SOLTN | 2 stated |
| BT5 | 3 | 2 | refused: hidden constraint is its hidden term plus a constant | CON1 leaves g a function of y alone |
| BT6 | 5 | 2 | refused: hidden constraint is its hidden term plus a constant | CON1 leaves g a function of y alone |
| BT7 | 5 | 3 | refused: hidden constraint is its hidden term plus a constant | CON1 leaves g a function of y alone |
| BT8 | 5 | 2 | refused: ill-conditioned split | the states' Jacobian block has condition number inf |
| BT9 | 4 | 2 | generated: [bt9.py](bt9.py) | hides CON1 |
| BYRDSPHR | 3 | 2 | refused: hidden constraint is its hidden term plus a constant | SPH2 leaves g a function of y alone |
| DIXCHLNG | 10 | 5 | refused: hidden constraint is its hidden term plus a constant | P6 leaves g a function of y alone |
| EXTRASIM | 2 | 1 | refused: only linear constraints | no element sum to hide |
| FLT | 2 | 2 | refused: no degrees of freedom | n = 2, m = 2 |
| GENHS28 | 10 | 8 | refused: only linear constraints | no element sum to hide |
| HONG | 4 | 1 | refused: only linear constraints | no element sum to hide |
| HS100LNP | 7 | 2 | generated: [hs100lnp.py](hs100lnp.py) | hides C1 |
| HS107 | 9 | 6 | refused: hidden constraint is its hidden term plus a constant | C3 leaves g a function of y alone |
| HS111 | 10 | 3 | refused: hidden constraint is its hidden term plus a constant | CON1 leaves g a function of y alone |
| HS112 | 10 | 3 | refused: only linear constraints | no element sum to hide |
| HS26 | 3 | 1 | refused: hidden constraint is its hidden term plus a constant | CON1 leaves g a function of y alone |
| HS27 | 3 | 1 | generated: [hs27.py](hs27.py) | hides CON1 |
| HS28 | 3 | 1 | refused: only linear constraints | no element sum to hide |
| HS39 | 4 | 2 | generated: [hs39.py](hs39.py) | hides CON1 |
| HS40 | 4 | 3 | refused: hidden constraint is its hidden term plus a constant | CON1 leaves g a function of y alone |
| HS41 | 4 | 1 | refused: only linear constraints | no element sum to hide |
| HS42 | 4 | 2 | refused: hidden constraint is its hidden term plus a constant | CON2 leaves g a function of y alone |
| HS46 | 5 | 2 | refused: hidden constraint is its hidden term plus a constant | CON1 leaves g a function of y alone |
| HS47 | 5 | 3 | refused: hidden constraint is its hidden term plus a constant | CON3 leaves g a function of y alone |
| HS48 | 5 | 2 | refused: only linear constraints | no element sum to hide |
| HS49 | 5 | 2 | refused: only linear constraints | no element sum to hide |
| HS50 | 5 | 3 | refused: only linear constraints | no element sum to hide |
| HS51 | 5 | 3 | refused: only linear constraints | no element sum to hide |
| HS52 | 5 | 3 | refused: only linear constraints | no element sum to hide |
| HS53 | 5 | 3 | refused: only linear constraints | no element sum to hide |
| HS54 | 6 | 1 | refused: only linear constraints | no element sum to hide |
| HS55 | 6 | 6 | refused: no degrees of freedom | n = 6, m = 6 |
| HS56 | 7 | 4 | generated: [hs56.py](hs56.py) | hides CON1 |
| HS6 | 2 | 1 | generated: [hs6.py](hs6.py) | hides G2 |
| HS60 | 3 | 1 | generated: [hs60.py](hs60.py) | hides C1 |
| HS61 | 3 | 2 | generated: [hs61.py](hs61.py) | hides C1 |
| HS62 | 3 | 1 | refused: only linear constraints | no element sum to hide |
| HS63 | 3 | 2 | refused: hidden constraint is its hidden term plus a constant | C2 leaves g a function of y alone |
| HS68 | ? | ? | refused: S2MPJ code can't be traced | AttributeError: 'SX' object has no attribute 'item' |
| HS69 | ? | ? | refused: S2MPJ code can't be traced | RuntimeError: .../casadi/core/sx_elem.cpp:165: Cannot compute the truth value of a CasADi SXElem symbolic expression. |
| HS7 | 2 | 1 | refused: hidden constraint is its hidden term plus a constant | CON1 leaves g a function of y alone |
| HS77 | 5 | 2 | generated: [hs77.py](hs77.py) | hides CON2 |
| HS78 | 5 | 3 | refused: hidden constraint is its hidden term plus a constant | C3 leaves g a function of y alone |
| HS79 | 5 | 3 | refused: hidden constraint is its hidden term plus a constant | C3 leaves g a function of y alone |
| HS8 | 2 | 2 | refused: no degrees of freedom | n = 2, m = 2 |
| HS80 | 5 | 3 | refused: hidden constraint is its hidden term plus a constant | C1 leaves g a function of y alone |
| HS81 | 5 | 3 | refused: hidden constraint is its hidden term plus a constant | C1 leaves g a function of y alone |
| HS87 | ? | ? | refused: S2MPJ code can't be traced | RuntimeError: .../casadi/core/sx_elem.cpp:165: Cannot compute the truth value of a CasADi SXElem symbolic expression. |
| HS9 | 2 | 1 | refused: only linear constraints | no element sum to hide |
| HS99 | 7 | 2 | refused: hidden constraint is its hidden term plus a constant | Q8E leaves g a function of y alone |
| LEWISPOL | 6 | 9 | refused: no degrees of freedom | n = 6, m = 9 |
| LIN | 4 | 2 | refused: only linear constraints | no element sum to hide |
| LSNNODOC | 5 | 4 | refused: only linear constraints | no element sum to hide |
| MARATOS | 2 | 1 | refused: hidden constraint is its hidden term plus a constant | C leaves g a function of y alone |
| MWRIGHT | 5 | 3 | refused: IPOPT disagrees with SOLTN | IPOPT found 24.97881, SOLTN is 32.851791437 |
| ODFITS | 10 | 6 | refused: only linear constraints | no element sum to hide |
| S316m322 | 2 | 1 | refused: no numeric SOLTN | none stated |
| STREGNE | 4 | 2 | generated: [stregne.py](stregne.py) | hides G1 |
| SUPERSIM | 2 | 2 | refused: no degrees of freedom | n = 2, m = 2 |
| TAME | 2 | 1 | refused: only linear constraints | no element sum to hide |
| TRYmB | 2 | 1 | refused: no numeric SOLTN | none stated |
| WACHBIEG | 3 | 2 | refused: IPOPT disagrees with SOLTN | IPOPT found 0.5, SOLTN is 1.0 |
