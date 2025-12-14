# RALEC-GNN

**Regime-Adaptive Learned Edge Construction for Crisis Detection**

A Graph Neural Network that learns dynamic cross-asset contagion networks to detect financial crises, challenging the Markowitz assumption that correlations remain stable.

## Results

| Model | Accuracy | Crisis Recall | Macro F1 | ROC-AUC |
|-------|----------|---------------|----------|---------|
| **RALEC-GNN** | **74.85%** | **88.24%** | **0.750** | **0.878** |
| Gradient Boosting | 62.94% | 66.33% | 0.532 | - |
| Random Forest | 59.29% | 55.93% | 0.504 | - |
| Logistic Regression | 58.37% | 41.67% | 0.465 | - |

## Quick Start

```bash
pip install -r requirements.txt
python run.py
```
