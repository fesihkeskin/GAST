## Imbalance Ratio ($\rho$) and Threshold ($\rho_{th}$)

This note mirrors the manuscript’s definition and decision rule in **Sec. Loss Function and Training Objective**.

### Definition

The class-imbalance ratio is computed as:

$$
\rho = \frac{\max_c N_c}{\min_c N_c},
$$

where $N_c$ is the number of labeled pixels in class $c$ (excluding padded labels and ignored/background labels).

### Threshold and loss-switch rule

The manuscript uses $\rho_{th}=5$ (selected via the ablation in Sec. *Ablation (Loss)*), with an adaptive loss:

- **Severe imbalance ($\rho > 5$):** use **Focal Loss**.
- **Moderate imbalance ($\rho \le 5$):** use **weighted Cross-Entropy (wCE)** with label smoothing.

## Dataset $\rho$ values used in the manuscript

The table below is aligned with the dataset-summary table in the manuscript (same $\rho$ values and the implied loss choice by the $\rho_{th}=5$ rule).

| Dataset | $\rho$ | Loss (by rule) |
|---|---:|---|
| Botswana | 1.50 | wCE / Cross-Entropy |
| Houston 2013 | 3.94 | wCE / Cross-Entropy |
| Indian Pines | 12.20 | Focal Loss |
| Kennedy Space Center (KSC) | 4.60 | wCE / Cross-Entropy |
| Pavia Centre | 24.61 | Focal Loss |
| Pavia University | 19.83 | Focal Loss |
| Salinas | 12.51 | Focal Loss |
| SalinasA | 4.00 | wCE / Cross-Entropy |

### Grouping by threshold

- **$\rho \le 5$ (wCE / Cross-Entropy):** Botswana, Houston 2013, KSC, SalinasA
- **$\rho > 5$ (Focal Loss):** Indian Pines, Pavia Centre, Pavia University, Salinas
