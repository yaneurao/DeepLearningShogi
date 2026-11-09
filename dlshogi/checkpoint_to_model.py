import argparse
import re
import sys
from collections import OrderedDict
from pathlib import Path

import torch

from dlshogi import serializers
from dlshogi.network.policy_value_network import policy_value_network


def infer_network(checkpoint_path):
    parent_name = checkpoint_path.resolve().parent.name
    network = re.sub(r"_round\d+$", "", parent_name)
    return network if network else None


def swa_module_state_dict(swa_state):
    module_state = OrderedDict()
    for key, value in swa_state.items():
        if key.startswith("module."):
            module_state[key[len("module.") :]] = value
    if module_state:
        return module_state
    return swa_state


def checkpoint_model_state(checkpoint, use_swa):
    if use_swa:
        if "swa_model" not in checkpoint:
            raise KeyError("checkpoint has no 'swa_model'")
        return swa_module_state_dict(checkpoint["swa_model"])

    if "model" not in checkpoint:
        raise KeyError("checkpoint has no 'model'")
    return checkpoint["model"]


def main(*argv):
    parser = argparse.ArgumentParser(
        description="Extract a dlshogi model file from a train.py .pth checkpoint."
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "model",
        type=Path,
        nargs="?",
        help="Output model path. Defaults to model-no_swa or model-swa next to the checkpoint.",
    )
    parser.add_argument(
        "--network",
        help="Network name. Defaults to the checkpoint parent directory name without _roundN.",
    )
    parser.add_argument(
        "--swa",
        action="store_true",
        help="Extract checkpoint['swa_model'] instead of checkpoint['model'].",
    )
    args = parser.parse_args(argv)

    checkpoint_path = args.checkpoint
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    network = args.network or infer_network(checkpoint_path)
    if not network:
        parser.error("--network is required when it cannot be inferred from the path")

    output_path = args.model
    if output_path is None:
        output_path = checkpoint_path.parent / ("model-swa" if args.swa else "model-no_swa")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint_model_state(checkpoint, args.swa)

    model = policy_value_network(network)
    model.load_state_dict(state_dict)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    serializers.save_npz(output_path, model)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main(*sys.argv[1:])
