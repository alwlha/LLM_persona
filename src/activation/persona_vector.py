from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import re
import warnings

import torch


TRAITS: tuple[str, ...] = (
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
)


MODEL_LAYER_HINTS = {
    "qwen": 18,
    "llama": 16,
}


def normalize_model_name(model_name: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", model_name.lower()).strip("-")
    return re.sub(r"-+", "-", normalized)


def resolve_trait_vector_path(path: str | Path, model_name: str) -> Path:
    raw_path = Path(path)
    if raw_path.name == "auto":
        auto_dir = raw_path.parent / normalize_model_name(model_name)
        if auto_dir.exists():
            return auto_dir
        raise ValueError(
            f"Cannot auto-resolve trait vectors for model '{model_name}' under: {raw_path.parent}"
        )

    model_dir = raw_path.parent / normalize_model_name(model_name)
    if raw_path.is_dir() and model_dir.exists() and model_dir != raw_path:
        warnings.warn(
            f"Auto-switching trait_vectors from '{raw_path}' to '{model_dir}' for model '{model_name}'",
            stacklevel=2,
        )
        return model_dir
    return raw_path


@dataclass
class PersonaSteeringSpec:
    layer: int
    vector: torch.Tensor
    positions: str = "all"


class ActivationSteeringAddition:
    """Lightweight activation-addition context manager for local HF models."""

    _POSSIBLE_LAYER_ATTRS: Iterable[str] = (
        "transformer.h",
        "encoder.layer",
        "model.layers",
        "language_model.layers",
        "gpt_neox.layers",
        "block",
    )

    def __init__(self, model: torch.nn.Module, layer_idx: int, vector: torch.Tensor, positions: str = "all"):
        self.model = model
        self.layer_idx = layer_idx
        self.vector = vector
        self.positions = positions
        self._handle = None

    def _locate_layer_list(self):
        for path in self._POSSIBLE_LAYER_ATTRS:
            cur = self.model
            for part in path.split("."):
                if hasattr(cur, part):
                    cur = getattr(cur, part)
                else:
                    break
            else:
                if hasattr(cur, "__getitem__"):
                    return cur
        raise ValueError("Cannot locate transformer layers in this model")

    def _hook_fn(self, module, ins, out):
        if torch.is_tensor(out):
            tensor_out = out
            tuple_mode = False
        elif isinstance(out, (tuple, list)) and torch.is_tensor(out[0]):
            tensor_out = out[0]
            tuple_mode = True
        else:
            return out

        steer = self.vector.to(tensor_out.device, dtype=tensor_out.dtype)
        if self.positions == "all":
            modified = tensor_out + steer
        else:
            modified = tensor_out.clone()
            modified[:, -1, :] += steer

        if tuple_mode:
            return (modified, *out[1:])
        return modified

    def __enter__(self):
        layers = self._locate_layer_list()
        if not (-len(layers) <= self.layer_idx < len(layers)):
            raise IndexError(f"layer {self.layer_idx} out of range (total={len(layers)})")
        self._handle = layers[self.layer_idx].register_forward_hook(self._hook_fn)
        return self

    def __exit__(self, *exc):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


def infer_target_layer(model_name: str, total_layers: int) -> int:
    model_lower = model_name.lower()
    for key, layer in MODEL_LAYER_HINTS.items():
        if key in model_lower:
            return min(layer, total_layers - 1)
    return total_layers // 2


def load_trait_vectors(path: str | Path, model_name: str) -> dict[str, torch.Tensor]:
    path = resolve_trait_vector_path(path, model_name=model_name)
    if path.is_dir():
        vectors: dict[str, torch.Tensor] = {}
        for trait in TRAITS:
            trait_path = path / f"{trait}.pt"
            if not trait_path.exists():
                raise ValueError(f"Missing trait vector file: {trait_path}")
            payload = torch.load(trait_path, map_location="cpu", weights_only=False)
            vectors[trait] = payload["vector"] if isinstance(payload, dict) and "vector" in payload else payload
        return vectors

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "trait_vectors" in payload:
        payload = payload["trait_vectors"]
    if not isinstance(payload, dict):
        raise ValueError("Trait vector file format invalid, expected a dict")

    missing = [trait for trait in TRAITS if trait not in payload]
    if missing:
        raise ValueError(f"Missing traits in vectors: {missing}")
    return payload


def combine_big5_vectors(
    trait_vectors: dict[str, torch.Tensor],
    coefficients: dict[str, float],
    normalize_per_layer: bool = True,
) -> torch.Tensor:
    reference = next(iter(trait_vectors.values())).float()
    combined = torch.zeros_like(reference)

    for trait, coeff in coefficients.items():
        if coeff == 0.0:
            continue
        vector = trait_vectors[trait].float()
        if normalize_per_layer:
            vector = vector / vector.norm(dim=1, keepdim=True).clamp_min(1e-8)
        combined = combined + coeff * vector
    return combined


def build_persona_steering_spec(meta: dict, model_name: str, total_layers: int, hidden_size: int) -> PersonaSteeringSpec:
    positions = meta.get("positions", "all")
    if positions not in {"all", "last"}:
        raise ValueError("meta.positions must be 'all' or 'last'")

    if "combined_vector" in meta:
        payload = torch.load(Path(meta["combined_vector"]), map_location="cpu", weights_only=False)
        if isinstance(payload, dict):
            if "vector" in payload:
                combined = payload["vector"]
            elif "axis" in payload:
                combined = payload["axis"]
            else:
                raise ValueError("combined_vector .pt dict must contain 'vector' or 'axis'")
        else:
            combined = payload
    else:
        if "trait_vectors" not in meta:
            raise ValueError("meta.trait_vectors or meta.combined_vector is required for vector activation")
        coefficients = {trait: float(meta.get("coefficients", {}).get(trait, 0.0)) for trait in TRAITS}
        if all(v == 0.0 for v in coefficients.values()):
            raise ValueError("All Big Five coefficients are 0.0")
        trait_vectors = load_trait_vectors(meta["trait_vectors"], model_name=model_name)
        combined = combine_big5_vectors(
            trait_vectors=trait_vectors,
            coefficients=coefficients,
            normalize_per_layer=bool(meta.get("normalize_per_layer", True)),
        )

    if combined.ndim != 2:
        raise ValueError(f"Combined vector must be 2D [layers, hidden], got shape={tuple(combined.shape)}")
    if combined.shape[0] != total_layers:
        raise ValueError(f"Vector layers mismatch: got {combined.shape[0]}, expected {total_layers}")
    if combined.shape[1] != hidden_size:
        raise ValueError(f"Vector hidden size mismatch: got {combined.shape[1]}, expected {hidden_size}")

    layer = int(meta.get("layer", infer_target_layer(model_name=model_name, total_layers=total_layers)))
    if not (-total_layers <= layer < total_layers):
        raise ValueError(f"Invalid layer index {layer} for total_layers={total_layers}")

    global_scale = float(meta.get("global_scale", 1.0))
    steer_vector = combined[layer].float() * global_scale
    return PersonaSteeringSpec(layer=layer, vector=steer_vector, positions=positions)
