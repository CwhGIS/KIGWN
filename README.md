# KIGWN

Reference implementation of **KIGWN** (*Knowledge-Informed Graph WaveNet*), the
model described in *Knowledge-Informed Graph Neural Networks for Spatiotemporal
Hydraulic State Prediction in Urban Drainage Systems*.

KIGWN predicts node-level hydraulic head and overflow one time step ahead over a
drainage-network graph. It combines a Graph WaveNet backbone with three
complementary elements:

| Component | Role |
| --- | --- |
| Multi-view graph supports | conduit topology, spatial proximity, learned data-driven adjacency |
| Multi-task learning | hydraulic head as an auxiliary target for the overflow branch |
| Process-informed overflow branch | weir/orifice descriptors of head exceedance, plus occurrence-guided (hurdle) gating |

This repository contains the **model definition and its architecture
configuration** only. The data pipeline, the training loop, the trained weights,
and the per-node normalisation statistics are not included.

## Repository contents

```
KIGWN/
├── __init__.py           public exports
├── config.json           architecture, input/output contract, training hyper-parameters
├── model_config.py       config loading and validation
├── requirements.txt      runtime dependency
├── graph_supports.py     topology and spatial-proximity (RBF) supports
├── graph_components.py   adaptive adjacency, random-walk normalisation, diffusion convolution
├── gwn_block.py          gated causal TCN + diffusion + residual/skip (one GWN block)
├── process_branches.py   physics descriptors, hurdle branch, overflow branch
├── kigwn_fullview.py     full model assembly and forward pass
├── losses.py             MSE / focal losses and the combined objective (Eq. 15)
├── smoke_test.py         contract and regression checks
└── README.md
```

## Requirements

- Python 3.9 or newer
- PyTorch 2.6.0 (CPU or CUDA; `config.json` records the version used in the paper)

No other third-party packages are required by the model code:

```bash
pip install -r requirements.txt
```

## Quick start

Place this directory on the import path, then build the model and run one
forward pass:

```python
import torch
from KIGWN import build_final_kigwn, build_rbf_support, build_topology_support

n_nodes, in_channels, window = 392, 9, 6

# Per-node training statistics and hydraulic attributes, in the same node order
# as x and the graph supports. Metres, not normalised values.
model = build_final_kigwn(
    head_min_per_node=torch.zeros(n_nodes),        # min of the head targets per node
    head_range_per_node=torch.ones(n_nodes),       # max - min of the head targets
    invert_elev_per_node=torch.zeros(n_nodes),     # conduit invert elevation
    max_depth_per_node=torch.full((n_nodes,), 3.0),  # node maximum depth
    outfall_mask_per_node=torch.zeros(n_nodes),    # 1 at outfalls; pass the checkpoint values
)
model.eval()

# x: [batch, 9, nodes, 6]; channel order and normalisation are in config.json.
x = torch.randn(2, in_channels, n_nodes, window)

# Static supports, each [nodes, nodes].
conduits = [(i, (i + 1) % n_nodes) for i in range(n_nodes)]        # illustrative
topology = build_topology_support(n_nodes, conduits)
rbf = build_rbf_support(torch.rand(n_nodes, 2, dtype=torch.float64) * 2000.0)

with torch.no_grad():
    out = model(x, [topology, rbf])

print(out["pred_head"].shape)   # torch.Size([2, 1, 392])
print(out["overflow_hard"].shape)  # torch.Size([2, 1, 392]), the reported-metric output
```

`build_final_kigwn()` reads the packaged `config.json` by default. Pass
`config_path=` for another file, or `config=` for a mapping (a full config
document or only its `architecture` section). Keyword overrides are applied on
top of the config and unknown keys are rejected.
