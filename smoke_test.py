"""Execution checks for the full-view KIGWN package.

Run with ``python -m KIGWN`` from the directory containing ``KIGWN``, or with
``python smoke_test.py``. The checks cover the config contract, the graph-support
semantics, the diffusion operator, and the forward/loss contract. They raise
``AssertionError`` with the failing invariant instead of relying on bare
``assert`` statements, so they also fail under ``python -O``.
"""

from __future__ import annotations

import copy
import json
import sys
import tempfile
from pathlib import Path

import torch

if __package__ in {None, ""}:  # Allow ``python smoke_test.py`` as well as -m.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from KIGWN import (
    build_final_kigwn,
    build_rbf_support,
    build_topology_support,
    kigwn_total_loss,
    load_model_config,
)
from KIGWN.graph_components import DiffusionConv
from KIGWN.losses import occurrence_focal_loss
from KIGWN.model_config import normalize_model_config
from KIGWN.process_branches import physics_features

_EXPECTED_PARAMETERS = 434_627
_CONFIG_PATH = Path(__file__).with_name("config.json")


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _expect(exc_type: type, function, fragment: str) -> None:
    try:
        function()
    except exc_type as error:
        _check(fragment in str(error), f"expected a message containing {fragment!r}, got: {error}")
    else:
        raise AssertionError(f"expected {exc_type.__name__} containing {fragment!r}, nothing raised")


def _physics_metadata(n_nodes: int) -> dict[str, torch.Tensor]:
    return {
        "head_min_per_node": torch.zeros(n_nodes),
        "head_range_per_node": torch.ones(n_nodes),
        "invert_elev_per_node": torch.zeros(n_nodes),
        "max_depth_per_node": torch.ones(n_nodes),
    }


def _check_config_contract() -> None:
    config = load_model_config()
    _check(config["n_nodes"] == 392, "shipped config should describe the 392-node network")
    _check(config["window_steps"] == 6, "shipped config should use a six-step input window")

    payload = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))

    tampered = copy.deepcopy(payload)
    tampered["input"]["feature_order"] = ["rainfall"] * 9
    _expect(ValueError, lambda: normalize_model_config(tampered), "feature_order")

    tampered = copy.deepcopy(payload)
    tampered["input"]["normalization"]["slope"] = "minmax"
    _expect(ValueError, lambda: normalize_model_config(tampered), "normalization")

    # Views must be validated even when the optional input block is absent.
    tampered = copy.deepcopy(payload)
    tampered.pop("input")
    tampered["views"] = ["topology"]
    _expect(ValueError, lambda: normalize_model_config(tampered), "views")

    with tempfile.TemporaryDirectory() as tmp:
        bad_path = Path(tmp) / "bad.json"
        bad_path.write_text(json.dumps(tampered), encoding="utf-8")
        _expect(ValueError, lambda: load_model_config(bad_path), "views")
        _expect(
            FileNotFoundError,
            lambda: load_model_config(Path(tmp) / "missing.json"),
            "Cannot read model config",
        )
        _expect(
            ValueError,
            lambda: build_final_kigwn(config_path=_CONFIG_PATH, config=payload),
            "not both",
        )

    # views must name exactly the supports the architecture builds.
    tampered = copy.deepcopy(payload)
    tampered["architecture"]["n_static_supports"] = 1
    _expect(ValueError, lambda: normalize_model_config(tampered), "views")

    # physics_features must name exactly the descriptors the model builds.
    tampered = copy.deepcopy(payload)
    tampered["physics_features"] = ["fill_ratio", "safety_margin"]
    _expect(ValueError, lambda: normalize_model_config(tampered), "physics_features")

    # in_channels must match the documented channel list.
    tampered = copy.deepcopy(payload)
    tampered["architecture"]["in_channels"] = 10
    _expect(ValueError, lambda: normalize_model_config(tampered), "in_channels")


def _check_graph_semantics(n_nodes: int) -> None:
    directed = build_topology_support(n_nodes, [(0, 1)])
    _check(directed[0, 1] == 1.0, "a conduit must map upstream -> downstream")
    _check(directed[1, 0] == 0.0, "the conduit direction must be preserved by default")
    _check(torch.count_nonzero(torch.diagonal(directed)) == 0, "self loops are off by default")

    symmetric = build_topology_support(n_nodes, [(0, 1)], symmetrize=True, self_loops=True)
    _check(symmetric[1, 0] == 1.0 and symmetric[0, 0] == 1.0, "symmetrize/self_loops must apply")

    knn = build_rbf_support(
        torch.tensor([[-100.0, 0.0], [0.0, 0.0], [100.0, 0.0]], dtype=torch.float64),
        max_radius=150.0,
        max_neighbors=1,
        min_neighbors=1,
    )
    _check(torch.equal(knn, knn.t()), "the spatial support must be undirected")
    _check(torch.count_nonzero(knn[1]) == 2, "symmetrisation must raise the degree above max_neighbors")
    _check(torch.count_nonzero(torch.diagonal(knn)) == 0, "the spatial support has no self loops")

    # Eq. (7) as printed: nothing beyond max_radius.
    line = torch.tensor([[0.0, 0.0], [50.0, 0.0], [10000.0, 0.0]], dtype=torch.float64)
    strict = build_rbf_support(line, max_radius=100.0, min_neighbors=0)
    _check(strict[2].sum() == 0, "min_neighbors=0 must enforce the strict radius cutoff")
    _check(strict[0, 1] > 0, "in-radius pairs must be weighted")

    # Documented isolation fallback: the far node keeps a beyond-radius neighbour.
    fallback = build_rbf_support(line, max_radius=100.0, min_neighbors=1)
    _check(fallback[2, 1] > 0, "the isolation fallback must keep a neighbour for a remote node")

    # sigma is the median of the per-node selections (100, 100, 150, 750 -> 125),
    # not of the unioned undirected edges (100, 150, 750 -> 150).
    asym = torch.tensor(
        [[0.0, 0.0], [100.0, 0.0], [250.0, 0.0], [1000.0, 0.0]], dtype=torch.float64
    )
    asym_support = build_rbf_support(asym, max_radius=400.0, max_neighbors=1, min_neighbors=1)
    expected = torch.exp(torch.tensor(-(100.0**2) / (2.0 * 125.0**2)))
    _check(
        torch.isclose(asym_support[0, 1], expected, rtol=0.0, atol=1e-7),
        "sigma must be the median over the per-node selections before symmetrisation",
    )

    probe = torch.tensor(
        [[0.0, 0.0], [100.0, 0.0], [0.0, 100.0], [100.0, 100.0]], dtype=torch.float64
    )
    offset = torch.tensor([428089.72, 4421368.74], dtype=torch.float64)
    _check(
        torch.allclose(
            build_rbf_support(probe),
            build_rbf_support(probe + offset),
            atol=1e-6,
            rtol=0.0,
        ),
        "centering must make the support translation invariant",
    )
    _expect(
        ValueError,
        lambda: build_rbf_support(probe.float() + offset.float()),
        "float64 input",
    )
    _expect(ValueError, lambda: build_rbf_support(probe, max_neighbors=0), "max_neighbors")


def _check_diffusion() -> None:
    support = torch.tensor(
        [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    )
    probe_x = torch.tensor([[[[1.0], [2.0], [4.0]]]])  # [B, C, N, T]

    def select(channel: int, k: int = 2) -> torch.Tensor:
        conv = DiffusionConv(c_in=1, c_out=1, k=k, n_supports=1)
        with torch.no_grad():
            conv.mlp.weight.zero_()
            conv.mlp.bias.zero_()
            conv.mlp.weight[0, channel, 0, 0] = 1.0
        return conv(probe_x, [support])

    # Channels are ordered [P_f^1, P_b^1, P_f^2, P_b^2, ...].
    _check(
        torch.equal(select(0), torch.tensor([[[[0.0], [1.0], [0.0]]]])),
        "forward diffusion must aggregate along the conduit direction",
    )
    _check(
        torch.equal(select(1), torch.tensor([[[[2.0], [0.0], [0.0]]]])),
        "reverse diffusion must aggregate against the conduit direction",
    )
    _check(
        torch.equal(select(2), torch.zeros(1, 1, 3, 1)),
        "the second propagation step must use the first-order state",
    )
    _expect(
        ValueError,
        lambda: DiffusionConv(c_in=1, c_out=1, k=2, n_supports=1)(probe_x, [support, support]),
        "graph supports",
    )


def _check_physics_features() -> None:
    """Pin the four descriptors to the formulas documented in process_branches."""

    depth = torch.full((1, 1, 5, 1), 2.0)
    invert = torch.zeros(1, 1, 5, 1)
    head = torch.tensor([[[[0.0], [1.0], [4.0], [-5.0], [100.0]]]])
    features = physics_features(head, invert, depth)
    _check(tuple(features.shape) == (1, 4, 5, 1), f"expected four descriptors, got {tuple(features.shape)}")

    fill_ratio, safety_margin, orifice, weir = (features[:, index, :, 0] for index in range(4))

    def close(actual: torch.Tensor, expected: list[float], label: str) -> None:
        reference = torch.tensor(expected).view_as(actual)
        _check(
            torch.allclose(actual, reference, atol=1e-6, rtol=0.0),
            f"{label}: got {actual.flatten().tolist()}, expected {expected}",
        )

    # fill_ratio is (H - z) / D clamped to [0, 10], so the dry node floors at 0
    # and the 50-depth node saturates at 10.
    close(fill_ratio, [0.0, 0.5, 2.0, 0.0, 10.0], "fill_ratio")
    close(safety_margin, [1.0, 0.5, 0.0, 1.0, 0.0], "safety_margin")

    # e = relu(fill_ratio - 1), i.e. the exceedance is depth-normalised.
    eps = 1e-6
    root = eps**0.5
    close(orifice, [0.0, 0.0, (1.0 + eps) ** 0.5 - root, 0.0, (9.0 + eps) ** 0.5 - root], "orifice")
    close(weir, [0.0, 0.0, 1.0, 0.0, 9.0**1.5], "weir")

    # Outfalls carry MaxDepth = 0: the divisor is clamped to eps, so any head
    # above the invert saturates fill_ratio and a dry outfall floors at zero.
    zero_depth = physics_features(head, invert, torch.zeros(1, 1, 5, 1))
    close(zero_depth[:, 0, :, 0], [0.0, 10.0, 10.0, 0.0, 10.0], "fill_ratio at zero depth")


def _check_model_contract() -> None:
    config = load_model_config()
    n_nodes = config["n_nodes"]

    _expect(ValueError, build_final_kigwn, "head_min_per_node")
    _expect(
        ValueError,
        lambda: build_final_kigwn(**{**_physics_metadata(n_nodes), "head_range_per_node": torch.zeros(n_nodes)}),
        "head_range_per_node must be positive",
    )
    # head_exceedance is de-normalised from the same buffers, so the statistics
    # are validated even when the physics embedding is disabled.
    _expect(
        ValueError,
        lambda: build_final_kigwn(
            **{
                **_physics_metadata(n_nodes),
                "use_physics_embedding": False,
                "head_range_per_node": torch.zeros(n_nodes),
            }
        ),
        "head_range_per_node must be positive",
    )
    _expect(
        ValueError,
        lambda: build_final_kigwn(**{**_physics_metadata(n_nodes), "outfall_mask_per_ndoe": torch.zeros(n_nodes)}),
        "Unknown KIGWN arguments",
    )

    torch.manual_seed(0)
    model = build_final_kigwn(**_physics_metadata(n_nodes))
    n_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    _check(
        n_parameters == _EXPECTED_PARAMETERS,
        f"architecture fingerprint changed: {n_parameters} trainable parameters "
        f"(expected {_EXPECTED_PARAMETERS}); update _EXPECTED_PARAMETERS if the change is intended",
    )

    topology = build_topology_support(n_nodes, [(index, (index + 1) % n_nodes) for index in range(n_nodes)])
    rbf = build_rbf_support(torch.rand(n_nodes, 2) * 2000.0)
    x = torch.randn(1, config["in_channels"], n_nodes, config["window_steps"])

    model.eval()
    with torch.no_grad():
        outputs = model(x, [topology, rbf])

    expected_shapes = {
        "pred_head": (1, 1, n_nodes),
        "prob_flood": (1, 1, n_nodes),
        "overflow_mag": (1, 1, n_nodes),
        "overflow_with_prob": (1, 1, n_nodes),
        "overflow_mask": (1, 1, n_nodes),
        "overflow_hard": (1, 1, n_nodes),
        "head_exceedance": (1, 1, n_nodes),
        "shared": (1, config["c_skip"], n_nodes, config["window_steps"]),
    }
    _check(set(outputs) == set(expected_shapes), f"unexpected output keys: {sorted(outputs)}")
    for key, shape in expected_shapes.items():
        _check(tuple(outputs[key].shape) == shape, f"{key} has shape {tuple(outputs[key].shape)}, expected {shape}")
        _check(torch.isfinite(outputs[key]).all(), f"{key} contains non-finite values")

    zeros = torch.zeros(1, 1, n_nodes)
    terms = kigwn_total_loss(outputs, zeros, zeros, zeros)
    _check(torch.isfinite(terms["total"]), "the total loss must be finite")
    _check(
        torch.allclose(terms["total"], terms["overflow"] + terms["head"] + terms["cls"]),
        "the total loss must be the sum of the three terms",
    )
    _check(
        torch.equal(terms["overflow"], torch.zeros(())),
        "with no overflow-positive cell the magnitude term must be zero, not the head loss",
    )
    weighted = kigwn_total_loss(outputs, zeros, zeros, zeros, task_weights=(2.0, 1.0, 1.0))
    _check(
        torch.allclose(weighted["total"], 2.0 * terms["head"] + terms["cls"] + terms["overflow"]),
        "task_weights must scale (head, cls, overflow) in that order",
    )
    saturated = occurrence_focal_loss(torch.tensor([0.0, 1.0]), torch.tensor([1.0, 0.0]))
    _check(
        torch.isfinite(saturated).all(),
        "the focal loss must stay finite at saturated probabilities",
    )

    # The magnitude term is an MSE over overflow-positive cells only.
    target = torch.zeros(1, 1, n_nodes)
    target[0, 0, :3] = torch.tensor([0.2, 0.5, 0.9])
    occurrence = (target > 0).float()
    terms = kigwn_total_loss(outputs, target, zeros, occurrence)
    positives = occurrence > 0
    _check(
        torch.allclose(
            terms["overflow"],
            torch.nn.functional.mse_loss(outputs["overflow_mag"][positives], target[positives]),
        ),
        "the overflow term must regress Q_raw on overflow-positive cells",
    )
    _check(
        torch.equal(outputs["overflow_mask"], (outputs["prob_flood"] >= 0.5).float()),
        "overflow_mask must be the 0.5 threshold of prob_flood",
    )
    _check(
        torch.equal(outputs["overflow_hard"], outputs["overflow_mask"] * outputs["overflow_mag"]),
        "overflow_hard must be the thresholded magnitude",
    )

    # Outfall nodes are marked with 1 and must not report overflow magnitude.
    masked_model = copy.deepcopy(model)
    with torch.no_grad():
        masked_model.outfall_mask_per_node[0] = 1.0
    with torch.no_grad():
        masked = masked_model(x, [topology, rbf])
    _check(
        bool((masked["overflow_mag"][0, 0, 0] == 0).all())
        and bool((masked["overflow_hard"][0, 0, 0] == 0).all()),
        "outfall nodes must have zero overflow magnitude",
    )
    _check(
        torch.equal(masked["overflow_mag"][0, 0, 1:], outputs["overflow_mag"][0, 0, 1:]),
        "the outfall mask must not change other nodes",
    )

    # The prediction is in normalised space: changing the de-normalisation
    # statistics must not change pred_head, but does change head_exceedance
    # (metres).
    rescaled = copy.deepcopy(model)
    with torch.no_grad():
        rescaled.head_min_per_node.fill_(100.0)
        rescaled.head_range_per_node.fill_(50.0)
        rescaled.invert_elev_per_node.fill_(10.0)
        rescaled.max_depth_per_node.fill_(3.0)
    with torch.no_grad():
        shifted = rescaled(x, [topology, rbf])
    _check(
        torch.equal(outputs["pred_head"], shifted["pred_head"]),
        "pred_head must stay in the normalised target space",
    )
    _check(
        not torch.allclose(outputs["head_exceedance"], shifted["head_exceedance"]),
        "head_exceedance must be expressed in metres",
    )
    # The descriptors reach both branches, so the occurrence probability depends
    # on the hydraulic metadata as well; see Section 3.5 of the manuscript.
    _check(
        not torch.allclose(outputs["prob_flood"], shifted["prob_flood"]),
        "the physics descriptors must also reach the occurrence branch",
    )
    _check(
        not torch.allclose(outputs["overflow_mag"], shifted["overflow_mag"]),
        "the physics descriptors must reach the magnitude branch",
    )

    # head_exceedance is de-normalised from the unclipped head: a prediction
    # outside [0, 1] must not be silently capped at the training range.
    probe = copy.deepcopy(model)
    with torch.no_grad():
        probe.head_predictor[-1].weight.zero_()
        probe.head_predictor[-1].bias.fill_(5.0)
        probe_outputs = probe(x, [topology, rbf])
    expected_exceedance = 5.0 - 1.0  # 5.0 m head, invert 0, max_depth 1
    _check(
        torch.allclose(probe_outputs["pred_head"], torch.full((1, 1, n_nodes), 5.0)),
        "the head probe must produce a constant prediction",
    )
    _check(
        torch.allclose(
            probe_outputs["head_exceedance"],
            torch.full((1, 1, n_nodes), expected_exceedance),
        ),
        "head_exceedance must be de-normalised from the unclipped head",
    )

    # Supports are cast to the model's device and dtype.
    with torch.no_grad():
        cast = model(x, [topology.double(), rbf.double()])
    _check(torch.equal(outputs["pred_head"], cast["pred_head"]), "supports must be cast to the model dtype")

    bad_support = topology.clone()
    bad_support[0, 1] = float("nan")
    _expect(ValueError, lambda: model(x, [bad_support, rbf]), "non-finite")
    _expect(ValueError, lambda: model(x.double(), [topology, rbf]), "dtype")

    # Ablation switches documented in config.json.
    without_hurdle = build_final_kigwn(**{**_physics_metadata(n_nodes), "use_hurdle": False}).eval()
    with torch.no_grad():
        no_gate = without_hurdle(x, [topology, rbf])
    _check(no_gate["prob_flood"] is None, "use_hurdle=False must not produce prob_flood")
    _expect(
        ValueError,
        lambda: kigwn_total_loss(
            no_gate,
            torch.zeros(1, 1, n_nodes),
            torch.zeros(1, 1, n_nodes),
            torch.zeros(1, 1, n_nodes),
        ),
        "use_hurdle=True",
    )

    without_adaptive = build_final_kigwn(**{**_physics_metadata(n_nodes), "use_adaptive": False}).eval()
    with torch.no_grad():
        without_adaptive(x, [topology, rbf])

    without_physics = build_final_kigwn(**{**_physics_metadata(n_nodes), "use_physics_embedding": False}).eval()
    _check(without_physics.physics_embedder is None, "the physics embedder must not be built when disabled")
    with torch.no_grad():
        without_physics(x, [topology, rbf])

    # The descriptors are detached, so the magnitude loss cannot train the head
    # predictor through them; the gate is trained by the occurrence loss alone.
    grad_model = build_final_kigwn(**_physics_metadata(n_nodes)).train()
    grad_outputs = grad_model(x, [topology, rbf])
    overflow_target = torch.zeros(1, 1, n_nodes)
    overflow_target[0, 0, :2] = 1.0
    terms = kigwn_total_loss(
        grad_outputs, overflow_target, zeros, (overflow_target > 0).float()
    )
    terms["overflow"].backward(retain_graph=True)
    _check(
        all(parameter.grad is None for parameter in grad_model.head_predictor.parameters()),
        "the magnitude loss must not reach head_predictor through the physics descriptors",
    )
    _check(
        any(parameter.grad is not None for parameter in grad_model.physics_embedder.parameters()),
        "the physics embedder must be trained by the magnitude loss",
    )
    _check(
        all(parameter.grad is None for parameter in grad_model.gate_branch.parameters()),
        "the magnitude loss must not train the occurrence branch",
    )
    grad_model.zero_grad(set_to_none=True)
    terms["cls"].backward()
    _check(
        any(parameter.grad is not None for parameter in grad_model.gate_branch.parameters()),
        "the occurrence loss must train the occurrence branch",
    )


def main() -> None:
    _check_config_contract()
    _check_graph_semantics(392)
    _check_diffusion()
    _check_physics_features()
    _check_model_contract()
    print("KIGWN smoke test passed")


if __name__ == "__main__":
    main()
