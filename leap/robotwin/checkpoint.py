"""Portable release checkpoints: plain configuration and tensor state dictionaries."""
from pathlib import Path
import torch
from leap.policy import LeaPPolicy


def save_checkpoint(path, policy, ema_policy, policy_config, optimizer, scheduler, epoch, step):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"format": "leap-release-v1", "policy_config": policy_config,
               "state_dicts": {"model": policy.state_dict(), "ema_model": ema_policy.state_dict()},
               "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
               "epoch": epoch, "step": step}
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_policy(path, device="cpu", use_ema=True):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != "leap-release-v1":
        raise ValueError("Expected a LeaP release checkpoint; research checkpoints need explicit conversion")
    policy = LeaPPolicy(**payload["policy_config"])
    policy.load_state_dict(payload["state_dicts"]["ema_model" if use_ema else "model"], strict=True)
    return policy.to(device).eval()
