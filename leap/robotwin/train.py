"""Standalone LeaP training on RoboTwin Zarr demonstrations."""
import argparse
import copy
import json
import math
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml
from leap.policy import LeaPPolicy
from leap.robotwin.checkpoint import save_checkpoint
from leap.robotwin.data import RobotDataset


@torch.no_grad()
def update_ema(ema, model, step):
    # Same warmup as research EMAModel: first two updates copy online weights.
    age = max(0, step - 1)
    decay = min(0.9999, 1 - (1 + age) ** -0.75) if age > 0 else 0.0
    for target, source in zip(ema.parameters(), model.parameters()):
        if source.requires_grad:
            target.mul_(decay).add_(source, alpha=1 - decay)
        else:
            target.copy_(source)
    for target, source in zip(ema.buffers(), model.buffers()):
        target.copy_(source)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--max-steps", type=int, help="Stop after this many updates (smoke test only)")
    args = parser.parse_args()
    with open(args.config) as stream:
        config = yaml.safe_load(stream)
    training = config["training"]
    epochs = args.epochs if args.epochs is not None else training["epochs"]
    batch_size = args.batch_size if args.batch_size is not None else training["batch_size"]
    if epochs < 1 or batch_size < 1 or (args.max_steps is not None and args.max_steps < 1):
        parser.error("epochs, batch-size and max-steps must be positive")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    policy_config = dict(config["policy"])
    dataset = RobotDataset(args.data, horizon=policy_config["horizon"],
                           n_obs_steps=policy_config["n_obs_steps"],
                           n_action_steps=policy_config["n_action_steps"], episodes=args.episodes,
                           val_ratio=training.get("val_ratio", 0.02), seed=args.seed)
    policy_config["shape_meta"] = dataset.shape_meta
    model = LeaPPolicy(**policy_config)
    model.set_normalizer(dataset.get_normalizer())
    model.to(args.device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=args.workers,
                        pin_memory=str(args.device).startswith("cuda"))
    val_loader = DataLoader(dataset.get_validation_dataset(), batch_size=batch_size,
                            shuffle=False, num_workers=args.workers)
    optimizer = torch.optim.AdamW(model.parameters(), **config["optimizer"])
    total_steps = epochs * len(loader)
    warmup = training["warmup_steps"]

    def lr_factor(step):
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    with (output / "run.json").open("w") as stream:
        json.dump({"args": vars(args), "config": config, "policy_config": policy_config,
                   "training_windows": len(dataset)}, stream, indent=2)
    step = 0
    with (output / "metrics.jsonl").open("w") as log:
        for epoch in range(epochs):
            model.train()
            for batch in loader:
                batch = {"obs": {k: v.to(args.device) for k, v in batch["obs"].items()},
                         "action": batch["action"].to(args.device)}
                optimizer.zero_grad(set_to_none=True)
                loss, metrics = model.compute_loss(batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss at step {step}")
                loss.backward()
                # Check finiteness without clipping by default, matching research train.py.
                limit = training.get("max_grad_norm")
                torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf") if limit is None else limit,
                                               error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                update_ema(ema, model, step)
                step += 1
                row = dict(metrics, loss=float(loss.detach()), epoch=epoch, step=step,
                           lr=optimizer.param_groups[0]["lr"])
                log.write(json.dumps(row) + "\n")
                log.flush()
                if step == 1 or step % 100 == 0:
                    print(json.dumps(row), flush=True)
                if args.max_steps and step >= args.max_steps:
                    break
            if len(val_loader) and epoch % training.get("val_every", 50) == 0:
                ema.eval()
                losses = []
                with torch.no_grad():
                    for batch in val_loader:
                        batch = {"obs": {k: v.to(args.device) for k, v in batch["obs"].items()},
                                 "action": batch["action"].to(args.device)}
                        val_loss, _ = ema.compute_loss(batch)
                        losses.append(float(val_loss))
                        if args.max_steps:
                            break
                with (output / "validation.jsonl").open("a") as validation:
                    validation.write(json.dumps({"epoch": epoch, "loss": float(np.mean(losses))}) + "\n")
            save_checkpoint(output / "checkpoints/latest.ckpt", model, ema, policy_config,
                            optimizer, scheduler, epoch, step)
            if epoch % training["checkpoint_every"] == 0 or epoch == epochs - 1:
                save_checkpoint(output / f"checkpoints/epoch={epoch:04d}.ckpt", model, ema,
                                policy_config, optimizer, scheduler, epoch, step)
            if args.max_steps and step >= args.max_steps:
                break
    print(f"Saved {output / 'checkpoints/latest.ckpt'} ({step} updates)", flush=True)


if __name__ == "__main__":
    main()
