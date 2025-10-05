| Dataset | Configuration | Val F1 | Val Brier | Test F1 | Test Brier |
| --- | --- | --- | --- | --- | --- |
| FEVER | Transformer | 0.715 | 0.207 | 0.756 | 0.183 |
| FEVER | No cross-attention | 0.014 | 0.253 | 0.050 | 0.251 |
| FEVER | No phase | 0.585 | 0.194 | 0.690 | 0.167 |
| FEVER | Feature dim = 16 | 0.665 | 0.183 | 0.710 | 0.177 |
| FEVER | MLP baseline | 0.667 | 0.181 | 0.712 | 0.181 |
| SciFact | Curriculum | 0.690 | 0.329 | 0.637 | 0.342 |
| SciFact | Finetune | 0.578 | 0.245 | 0.519 | 0.239 |
| HoVer | FEVER adapt | 0.095 | 0.480 | - | - |
| HoVer | FEVER base | 0.000 | 0.516 | - | - |
| HoVer | Transformer | 0.000 | 0.516 | 0.000 | 0.526 |
