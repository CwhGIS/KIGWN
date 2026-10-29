"""Configuration loading and validation for the KIGWN model.

Two public entry points share the same validation:

* :func:`load_model_config` reads a JSON file (``config.json`` by default) and
  also checks the descriptive ``input``/``views`` metadata against the
  executable architecture;
* :func:`normalize_model_config` accepts a Python mapping (full document or
  architecture section only) and performs the same checks for whatever
  metadata is present.

The ``_ARTICLE_*`` constants below mirror ``config.json``; they are kept in the
module so that a metadata edit in the JSON cannot silently drift from the
feature layout and the physics descriptors the model expects. Update both
together.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


_DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.json")
_ARCHITECTURE_KEYS = frozenset(
    {
        "n_nodes",
        "in_channels",
        "c_res",
        "c_skip",
        "diffusion_steps",
        "dilations",
        "use_adaptive",
        "use_physics_embedding",
        "use_hurdle",
        "n_static_supports",
        "physics_embed_dim",
        "adaptive_rank",
        "window_steps",
    }
)
_POSITIVE_INT_KEYS = (
    "n_nodes",
    "in_channels",
    "c_res",
    "c_skip",
    "diffusion_steps",
    "n_static_supports",
    "physics_embed_dim",
    "adaptive_rank",
    "window_steps",
)
_BOOL_KEYS = ("use_adaptive", "use_physics_embedding", "use_hurdle")
_ARTICLE_FEATURE_ORDER = (
    "rainfall",
    "cumulative_rainfall",
    "area",
    "imperviousness",
    "width",
    "slope",
    "invert_elevation",
    "max_depth",
    "node_type",
)
_ARTICLE_VIEWS = ("topology", "rbf", "adaptive")
_ARTICLE_NORMALIZATION = {
    "rainfall": "log1p_minmax",
    "cumulative_rainfall": "minmax",
    "area": "log1p_minmax",
    "imperviousness": "log1p_minmax",
    "width": "log1p_minmax",
    "slope": "log1p_minmax",
    "invert_elevation": "log1p_minmax",
    "max_depth": "log1p_minmax",
    "node_type": "none_0_1",
}
_ARTICLE_PHYSICS_FEATURES = (
    "fill_ratio",
    "safety_margin",
    "orifice_exceedance_sqrt",
    "weir_exceedance_power_1_5",
)


def _validate_architecture_keys(payload: Mapping[str, Any]) -> None:
    unknown = sorted(set(payload) - _ARCHITECTURE_KEYS)
    if unknown:
        raise ValueError(f"Unknown model config keys: {', '.join(unknown)}")
    missing = sorted(_ARCHITECTURE_KEYS - set(payload))
    if missing:
        raise ValueError(f"Missing model config keys: {', '.join(missing)}")


def _validate_scalar_values(values: Mapping[str, Any]) -> None:
    for key in _POSITIVE_INT_KEYS:
        value = values[key]
        if type(value) is not int or value <= 0:
            raise ValueError(f"{key} must be a positive integer.")
    for key in _BOOL_KEYS:
        if type(values[key]) is not bool:
            raise ValueError(f"{key} must be a boolean.")


def _normalise_dilations(values: dict[str, Any]) -> None:
    dilations = values["dilations"]
    if isinstance(dilations, (str, bytes)) or not isinstance(dilations, (list, tuple)):
        raise ValueError("dilations must be a non-empty list of positive integers.")
    if not dilations or any(type(value) is not int or value <= 0 for value in dilations):
        raise ValueError("dilations must be a non-empty list of positive integers.")
    values["dilations"] = tuple(dilations)


def _normalise_architecture(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one architecture mapping and return constructor-ready values."""

    if not isinstance(payload, Mapping):
        raise ValueError("Model config architecture must be a JSON object.")
    _validate_architecture_keys(payload)
    values = dict(payload)
    _validate_scalar_values(values)
    _normalise_dilations(values)
    return values


def _extract_architecture(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError("Model config must be a JSON object.")
    architecture = payload.get("architecture", payload)
    if not isinstance(architecture, Mapping):
        raise ValueError("Model config 'architecture' must be a JSON object.")
    return architecture


def _validate_input_spec(input_spec: Mapping[str, Any], architecture: Mapping[str, Any]) -> None:
    if input_spec.get("window_steps") != architecture["window_steps"]:
        raise ValueError("Input window_steps disagrees with architecture.window_steps.")
    if input_spec.get("horizon_steps") != 1:
        raise ValueError("horizon_steps must be 1.")
    feature_order = input_spec.get("feature_order")
    if not isinstance(feature_order, (list, tuple)) or tuple(feature_order) != _ARTICLE_FEATURE_ORDER:
        raise ValueError("input.feature_order does not match the expected nine channels.")
    if architecture["in_channels"] != len(_ARTICLE_FEATURE_ORDER):
        raise ValueError(
            f"in_channels must be {len(_ARTICLE_FEATURE_ORDER)}, the length of "
            "input.feature_order."
        )
    normalization = input_spec.get("normalization")
    if normalization != _ARTICLE_NORMALIZATION:
        raise ValueError("input.normalization does not match the expected per-channel rules.")


def _validate_view_metadata(payload: Mapping[str, Any], architecture: Mapping[str, Any]) -> None:
    """Check that ``views`` names exactly the supports the model will be given."""

    views = payload.get("views")
    if views is None:
        return
    n_supports = architecture["n_static_supports"] + int(architecture["use_adaptive"])
    if architecture["n_static_supports"] > len(_ARTICLE_VIEWS) - 1:
        raise ValueError(
            "n_static_supports must not exceed "
            f"{len(_ARTICLE_VIEWS) - 1}: the metadata names only the topology and "
            "spatial-proximity views."
        )
    expected = _ARTICLE_VIEWS[:n_supports]
    if not isinstance(views, (list, tuple)) or tuple(views) != expected:
        raise ValueError(
            "views must be " + ", ".join(expected) + " for "
            f"n_static_supports={architecture['n_static_supports']} and "
            f"use_adaptive={architecture['use_adaptive']}."
        )


def _validate_physics_features(payload: Mapping[str, Any]) -> None:
    """Check that ``physics_features`` names exactly the descriptors the model builds."""

    features = payload.get("physics_features")
    if features is None:
        return
    if not isinstance(features, (list, tuple)) or tuple(features) != _ARTICLE_PHYSICS_FEATURES:
        raise ValueError(
            "physics_features must be "
            + ", ".join(_ARTICLE_PHYSICS_FEATURES)
            + " in that order, the channel order returned by physics_features()."
        )


def _validate_input_metadata(payload: Mapping[str, Any], architecture: Mapping[str, Any]) -> None:
    """Check descriptive metadata against the executable architecture.

    ``views`` and ``physics_features`` are validated even when the optional
    ``input`` block is absent.
    """

    _validate_view_metadata(payload, architecture)
    _validate_physics_features(payload)
    input_spec = payload.get("input")
    if input_spec is None:
        return
    if not isinstance(input_spec, Mapping):
        raise ValueError("Model config 'input' must be a JSON object.")
    _validate_input_spec(input_spec, architecture)


def normalize_model_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalise a config mapping for ``KIGWNFullView``.

    Accepts the full config document or an architecture-only mapping. Any
    ``input``/``views``/``physics_features`` metadata that is present is checked
    exactly as in :func:`load_model_config`, so the two entry points cannot
    disagree.
    """

    architecture = _normalise_architecture(_extract_architecture(config))
    _validate_input_metadata(config, architecture)
    return architecture


def load_model_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load and validate the architecture from ``config.json``.

    Returns constructor arguments only; the descriptive ``input``, ``views`` and
    ``physics_features`` metadata in the file is checked but not passed into the
    model.

    Raises:
        FileNotFoundError: if the path does not exist.
        OSError: for any other read failure (permissions, a directory, ...).
        ValueError: for invalid JSON, an unsupported ``schema_version`` or
            ``model``, or inconsistent metadata.
    """

    path = _DEFAULT_CONFIG_PATH if config_path is None else Path(config_path)
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Cannot read model config: {path}") from exc
    except OSError as exc:
        raise OSError(f"Cannot read model config: {path}: {exc}") from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in model config: {path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("Model config must be a JSON object.")
    if payload.get("schema_version", 1) != 1:
        raise ValueError("Unsupported model config schema_version.")
    if payload.get("model", "KIGWN") != "KIGWN":
        raise ValueError("Model config must describe KIGWN.")
    return normalize_model_config(payload)


__all__ = ["load_model_config", "normalize_model_config"]
