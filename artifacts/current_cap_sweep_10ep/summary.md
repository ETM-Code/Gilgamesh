# Current-Cap Sweep (10 epochs)

This sweep varies per-synapse current cap and total current budget.

| Case | I_syn_max (uA) | I_total_max (uA) | I_total/I_syn | Best Acc (%) | Final Acc (%) | Runtime (s) |
|---|---:|---:|---:|---:|---:|---:|
| nocap_reference | 3.00 | 50.00 | 16.67 | 83.63 | 83.60 | 4.62 |
| syn_1p0_total_50 | 1.00 | 50.00 | 50.00 | 73.87 | 73.87 | 4.65 |
| syn_2p0_total_50 | 2.00 | 50.00 | 25.00 | 81.56 | 81.52 | 4.69 |
| syn_3p0_total_50 | 3.00 | 50.00 | 16.67 | 83.74 | 83.70 | 4.75 |
| syn_4p5_total_50 | 4.50 | 50.00 | 11.11 | 85.04 | 84.89 | 4.71 |
| syn_4p5_total_75 | 4.50 | 75.00 | 16.67 | 84.85 | 84.85 | 4.71 |
| syn_6p0_total_100 | 6.00 | 100.00 | 16.67 | 85.61 | 85.51 | 4.72 |
| syn_3p0_total_30 | 3.00 | 30.00 | 10.00 | 83.36 | 83.36 | 4.78 |

## Quick Read
- Best capped case: `syn_6p0_total_100` at 85.61% best test accuracy.
- Worst capped case: `syn_1p0_total_50` at 73.87% best test accuracy.
- Compare `syn_4p5_total_50` vs `syn_4p5_total_75` to isolate effect of total-current budget at fixed per-synapse cap.
- Compare `syn_3p0_total_50` vs `syn_3p0_total_30` to isolate tighter total-cap effect at nominal per-synapse cap.
