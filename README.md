# KIGWN

Knowledge-informed graph neural network for spatiotemporal hydraulic-state
prediction in urban drainage systems.

This repository provides the KIGWN model definition accompanying the manuscript
*Knowledge-Informed Graph Neural Networks for Spatiotemporal Hydraulic State
Prediction in Urban Drainage Systems*. KIGWN combines a Graph WaveNet backbone
with a multi-view graph representation, a multi-task formulation, and a
process-informed overflow branch. It predicts node-level hydraulic head and
overflow across a drainage network.

The package contains the model definition and its architecture configuration
(`config.json`). The data pipeline, the training loop, and the trained weights
are not included.

## Requirements

- Python 3.9+
- PyTorch 2.6.0

## Usage

Build the model, pass the two static graph supports, and run the forward path:

```python
import torch
from final_kigwn_minimal import build_final_kigwn

nodes = 392
# Per-node training statistics and physical attributes, in the same node order
# as x and both graph supports. Invert elevation and maximum depth are supplied
# in metres, not as normalized values.
model = build_final_kigwn(
    head_min_per_node=torch.zeros(nodes),
    head_range_per_node=torch.ones(nodes),
    invert_elev_per_node=torch.zeros(nodes),
    max_depth_per_node=torch.ones(nodes),
)
model.eval()

x = torch.randn(2, 9, nodes, 6)
topology = torch.eye(nodes)
rbf = torch.eye(nodes)

with torch.no_grad():
    outputs = model(x, [topology, rbf])
```

`x` is the input sequence with shape `[batch, 9, nodes, 6]`, where the nine
channels are rainfall, cumulative rainfall, and seven static node attributes in
the order listed in `config.json`. `static_supports` holds the two static graph
supports, each `[nodes, nodes]`; the third, data-driven support is constructed
inside the model. The package also provides `build_topology_support` and
`build_rbf_support` for constructing the two static supports.

The spatial-proximity and data-driven views correspond to the `rbf` and
`adaptive` entries recorded in `config.json`. The forward pass returns the
predicted hydraulic head, the overflow occurrence probability, and the
corresponding overflow magnitude.

Run the forward path on synthetic inputs with:

```bash
python -m final_kigwn_minimal
```
