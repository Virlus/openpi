import collections
import dataclasses
import logging
import re
from typing import Protocol, runtime_checkable

import flax.traverse_util
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download

logger = logging.getLogger(__name__)


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        # Add all missing LoRA weights.
        return _merge_params(loaded_params, params, missing_regex=".*lora.*")


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")


def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.

    Returns:
        A new dictionary with the merged parameters.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    # First, take all weights that are a subset of the reference weights.
    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            result[k] = v.astype(flat_ref[k].dtype) if v.dtype != flat_ref[k].dtype else v

    flat_loaded.clear()

    # Then, merge any missing weights as defined by the missing regex.
    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = flat_ref[k]

    return flax.traverse_util.unflatten_dict(result, sep="/")


@dataclasses.dataclass(frozen=True)
class Pi05ValueWeightLoader(WeightLoader):
    """Loads pi05 checkpoints while adding an additional value expert.

    This loader:
      * loads the original pi05 checkpoint,
      * remaps the old action expert (expert #1) to expert #2,
      * optionally initializes the new value expert (#1) from an external loader,
        defaulting to the official PaliGemma checkpoint.
    """

    base_params_path: str
    value_weight_loader: WeightLoader | None = dataclasses.field(default_factory=PaliGemmaWeightLoader)

    def load(self, params: at.Params) -> at.Params:
        flat_meta = flax.traverse_util.flatten_dict(params, sep="/")
        flat_result = dict(flat_meta)

        base_params = _model.restore_params(download.maybe_download(self.base_params_path), restore_type=np.ndarray)
        flat_base = flax.traverse_util.flatten_dict(base_params, sep="/")

        flat_value: dict[str, np.ndarray | at.Array] = {}
        if self.value_weight_loader is not None:
            value_params = self.value_weight_loader.load(params)
            flat_value = flax.traverse_util.flatten_dict(value_params, sep="/")

        value_key_map = _build_value_key_map(flat_meta.keys())
        action_key_map = _build_action_key_map(flat_meta.keys())

        def assign(target_key: str, source_value: np.ndarray):
            if target_key not in flat_result:
                return
            if not isinstance(source_value, np.ndarray):
                return
            reference = flat_meta[target_key]
            target_shape = getattr(reference, "shape", None)
            if target_shape is not None and source_value.shape != target_shape:
                logger.warning(
                    "Skipping %s due to shape mismatch (loaded %s vs expected %s).",
                    target_key,
                    source_value.shape,
                    target_shape,
                )
                return
            target_dtype = getattr(reference, "dtype", None)
            if target_dtype is not None and source_value.dtype != target_dtype:
                source_value = source_value.astype(target_dtype, copy=False)
            flat_result[target_key] = source_value

        for key, value in flat_base.items():
            if key in flat_result:
                assign(key, value)
            if key in action_key_map:
                assign(action_key_map[key], value)

        if value_key_map:
            for base_key, target_keys in value_key_map.items():
                source_value = flat_value.get(base_key)
                if source_value is None:
                    source_value = flat_base.get(base_key)
                if not isinstance(source_value, np.ndarray):
                    continue
                for target_key in target_keys:
                    assign(target_key, source_value)

        return flax.traverse_util.unflatten_dict(flat_result, sep="/")


def _build_value_key_map(keys: collections.abc.Iterable[str]) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = collections.defaultdict(list)
    for key in keys:
        normalized, replaced = _replace_expert_suffix(key, "_1", "")
        if replaced:
            mapping[normalized].append(key)
    return mapping


def _build_action_key_map(keys: collections.abc.Iterable[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for key in keys:
        normalized, replaced = _replace_expert_suffix(key, "_2", "_1")
        if replaced:
            mapping[normalized] = key
    return mapping


def _replace_expert_suffix(key: str, old_suffix: str, new_suffix: str) -> tuple[str, bool]:
    segments = key.split("/")
    try:
        llm_idx = segments.index("llm")
    except ValueError:
        return key, False
    replaced = False
    for idx in range(llm_idx + 1, len(segments)):
        segment = segments[idx]
        if segment.endswith(old_suffix):
            segments[idx] = segment[: -len(old_suffix)] + new_suffix
            replaced = True
    return "/".join(segments), replaced
