""" 
Original script by Chengyi Su, modified by Matt Stirling. 
"""
import argparse
from pathlib import Path
import json
import csv
import os
import random
import time

# import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from emonet.models import EmoNet
from emonet.data import EmoNetCSV
from tqdm import tqdm


#region HELPERS

# Utils: seeds, metrics, I/O
def set_seed(seed: int):
    # Make results more reproducible across Python, NumPy, and PyTorch (CPU+CUDA).
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def ccc(x, y, eps=1e-8):
    """
    Concordance Correlation Coefficient ([-1, 1]).
    Higher is better.
    """
    x = x.float(); y = y.float()
    xm, ym = x.mean(), y.mean()
    xv, yv = x.var(unbiased=False), y.var(unbiased=False) # variance
    cov = ((x - xm) * (y - ym)).mean() # PCC * SD(x) * SD(y)
    return (2 * cov) / (xv + yv + (xm - ym) ** 2 + eps)

def ccc_loss(x, y): return 1.0 - ccc(x, y)

def rmse(x, y): return torch.sqrt(torch.mean((x - y) ** 2))
def mae(x, y):  return torch.mean(torch.abs(x - y))

def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

# Canonical class orders used by EmoNet expression head.
CANONICAL_8 = ["neutral","happy","sad","surprise","fear","disgust","anger","contempt"]
CANONICAL_5 = ["neutral","happy","sad","surprise","fear"] 

def build_fixed_label_map(nclasses: int):
    if nclasses == 8:
        order = CANONICAL_8
    elif nclasses == 5:
        order = CANONICAL_5
    else:
        raise ValueError("nclasses must be 5 or 8")
    return {name: i for i, name in enumerate(order)}

def get_dataset_label2id(csv: str) -> dict|None:
    df_train = pd.read_csv(csv)
    label2id = None
    if args.use_expr:
        label2id = build_fixed_label_map(args.nclasses)
        # sanity-check dataset labels
        seen = set(df_train["label"].astype(str).str.strip().str.lower().unique())
        expected = set(label2id.keys())
        unknown = seen - expected
        if unknown:
            raise ValueError(
                f"Unknown labels in CSV: {sorted(unknown)}. "
                f"Expected one of: {sorted(expected)} (case-insensitive)."
            )
        print(f"Label map (fixed to EmoNet order): {label2id}")
    return label2id


#region EVAL & TRAIN

# Evaluation
def evaluate(model, loader, device, use_expr):
    """
    Run validation over a loader and compute VA metrics (CCC/RMSE/MAE),
    plus expression accuracy if enabled.
    """
    model.eval()
    v_pred, v_true, a_pred, a_true = [], [], [], []
    expr_correct, expr_total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            vp = out["valence"].view(-1).cpu()
            ap = out["arousal"].view(-1).cpu()
            v_pred.append(vp); a_pred.append(ap)
            v_true.append(y["valence"]); a_true.append(y["arousal"])
            if use_expr and "expression" in out and "expr" in y:
                pred = out["expression"].argmax(1).cpu()
                expr_correct += (pred == y["expr"]).sum().item()
                expr_total += pred.numel()
    v_pred = torch.cat(v_pred); v_true = torch.cat(v_true)
    a_pred = torch.cat(a_pred); a_true = torch.cat(a_true)

    metrics = {
        "ccc_v": ccc(v_pred, v_true).item(),
        "ccc_a": ccc(a_pred, a_true).item(),
        "rmse_v": rmse(v_pred, v_true).item(),
        "rmse_a": rmse(a_pred, a_true).item(),
        "mae_v": mae(v_pred, v_true).item(),
        "mae_a": mae(a_pred, a_true).item(),
    }
    metrics["ccc_mean"] = 0.5 * (metrics["ccc_v"] + metrics["ccc_a"])
    if use_expr and expr_total > 0:
        metrics["expr_acc"] = expr_correct / expr_total
    return metrics

# One training epoch
def train_one_epoch(model, loader, device, optimizer, scaler, use_expr, va_loss_func=ccc_loss, lambda_expr=1.0, epoch=1, epochs=1):
    model.train()
    ce = nn.CrossEntropyLoss()
    running = {"loss": 0.0, "loss_va": 0.0, "loss_expr": 0.0}
    n = 0

    bar = tqdm(loader, desc=f"Train Epoch {epoch}/{epochs}", leave=False)
    for x, y in bar:
        x = x.to(device, non_blocking=True)
        v = y["valence"].to(device)
        a = y["arousal"].to(device)

        with torch.amp.autocast('cuda', enabled=scaler is not None): # type: ignore
            out = model(x)

            # VA regression losses
            loss_v = va_loss_func(out["valence"].view(-1), v)
            loss_a = va_loss_func(out["arousal"].view(-1), a)
            loss_va = loss_v + loss_a
            loss = loss_va
            if use_expr and "expression" in out and "expr" in y:
                logits = out["expression"]
                loss_expr = ce(logits, y["expr"].to(device))
                loss = loss + lambda_expr * loss_expr
            else:
                loss_expr = torch.tensor(0.0, device=device)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        else:
            loss.backward(); optimizer.step()

        bs = x.size(0); n += bs
        running["loss"] += loss.item() * bs
        running["loss_va"] += loss_va.item() * bs
        running["loss_expr"] += loss_expr.item() * bs

        # live numbers on the bar
        bar.set_postfix({
            "loss": f"{loss.item():.3f}",
            "va": f"{loss_va.item():.3f}",
            "lr": f"{optimizer.param_groups[0]['lr']:.1e}"
        })

    # Normalize by total samples
    for k in running: running[k] /= max(1, n)
    return running


def unfreeze_backbone(model):
    unfrozen = 0
    for _, p in model.named_parameters():
        if not p.requires_grad:
            p.requires_grad = True
            unfrozen += 1
    print(f"Unfroze {unfrozen} params. Consider lowering LR.")


CONTINUOUS_LOSS_FUNCS = {
    "ccc_loss": ccc_loss,
    "mse":      nn.MSELoss,
    "mae":      nn.L1Loss,
    "hubert":   nn.HuberLoss,
}

#region MAIN
def main(args):

    va_loss_func = CONTINUOUS_LOSS_FUNCS.get(args.va_loss_func)
    if va_loss_func is None:
        print('ERROR: no such loss function available:', args.va_loss_func)
        return
    print(f'Using {args.va_loss_func} as loss function for valence-arousal')

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # dataset paths
    train_csv =  args.dataset_root + os.sep + args.train_csv
    test_csv =   args.dataset_root + os.sep + args.test_csv
    train_root = args.dataset_root + os.sep + args.train_folder
    test_root =  args.dataset_root + os.sep + args.test_folder

    # datasets
    label2id = get_dataset_label2id(train_csv)

    # Datasets & loaders
    train_ds = EmoNetCSV(train_csv, train_root, size=args.size,
                         use_expr=args.use_expr, label2id=label2id, augment=True)
    test_ds  = EmoNetCSV(test_csv,  test_root,  size=args.size,
                         use_expr=args.use_expr, label2id=label2id, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch, shuffle=False,
                              num_workers=4, pin_memory=True)

    # Model: initialize with desired expression classes 
    model = EmoNet(n_expression=args.nclasses).to(device)

    # load pretrained weights
    if args.pretrained_params is None:
        res = input("\nWarning: no pretrained parameters passed, train model from scratch? (y)\n> ")
        if res not in ["y", "yes", "Y"]:
            print("quitting")
            return
    
    else:
        params_file = Path(args.pretrained_params)
        if not params_file.exists():
            raise FileExistsError(f"given params file doesnt exist: {params_file}")
        print(f"Loading pretrained weights: {params_file}")
        state = torch.load(str(params_file), map_location="cpu")
        state = { k.replace("module.", ""): v for k, v in state.items() }
        model.load_state_dict(state, strict=False)

    # only train last layer by default
    trainable_params = [ p for p in model.parameters() if p.requires_grad ]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler('cuda') if (args.amp and device == "cuda") else None  # type: ignore

    # csv file
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    best, best_path = -1e9, outdir / "ckpt_best.pth"
    log_csv = outdir / "metrics.csv"
    with open(log_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["epoch","train_loss","train_va","train_expr",
                  "ccc_v","ccc_a","ccc_mean","rmse_v","rmse_a","mae_v","mae_a"]
        if args.use_expr: header.append("expr_acc")
        w.writerow(header)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Optional staged fine-tuning: unfreeze backbone at a chosen epoch
        if args.unfreeze_backbone_after > 0 and epoch == args.unfreeze_backbone_after:
            unfreeze_backbone(model)
            for g in optimizer.param_groups:
                g["lr"] = args.lr * 0.3             # Reduce LR to avoid destabilizing training after unfreezing
            print(f"Lowered LR to {optimizer.param_groups[0]['lr']} after unfreezing.")

        # Train + Validate
        tr = train_one_epoch(model, train_loader, device, optimizer, scaler,
                     use_expr=args.use_expr, lambda_expr=args.lambda_expr,
                     epoch=epoch, epochs=args.epochs)

        vl = evaluate(model, test_loader, device, use_expr=args.use_expr)

        # Console log
        dt = time.time() - t0
        msg = (f"[{epoch:03d}/{args.epochs}] "
               f"loss={tr['loss']:.4f} va_loss={tr['loss_va']:.4f} "
               f"| ccc(V)={vl['ccc_v']:.3f} ccc(A)={vl['ccc_a']:.3f} mean={vl['ccc_mean']:.3f} "
               f"rmse(V)={vl['rmse_v']:.3f} rmse(A)={vl['rmse_a']:.3f} "
               f"mae(V)={vl['mae_v']:.3f} mae(A)={vl['mae_a']:.3f} time={dt:.1f}s")
        if args.use_expr and "expr_acc" in vl:
            msg += f" | expr_acc={vl['expr_acc']:.3f}"
        print(msg)

        with open(log_csv, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            row = [epoch, tr["loss"], tr["loss_va"], tr["loss_expr"],
                   vl["ccc_v"], vl["ccc_a"], vl["ccc_mean"],
                   vl["rmse_v"], vl["rmse_a"], vl["mae_v"], vl["mae_a"]]
            if args.use_expr and "expr_acc" in vl: row.append(vl["expr_acc"])
            w.writerow(row)

        # Track and save the best checkpoint by mean CCC over V/A
        if vl["ccc_mean"] > best:
            best = vl["ccc_mean"]
            torch.save(model.state_dict(), best_path)
            print(f"  ↳ saved best to {best_path} (ccc_mean={best:.3f})")

    # Final export of last-epoch weights (best already saved separately)
    export = outdir / f"emonet_{args.nclasses}_finetuned.pth"
    torch.save(model.state_dict(), export)
    print(f"Done. Best ckpt: {best_path}\nExport: {export}")



#region CLI
if __name__ == "__main__":
    
    default_pretrained = Path(__file__).parent / "pretrained" / f"emonet_{8}.pth"
    
    ap = argparse.ArgumentParser()

    # data paths
    ap.add_argument("--pretrained_params", type=str)
    ap.add_argument("--dataset_root",  type=str, required=True,
                    help="path to dataset root")
    ap.add_argument("--train_csv",  type=str, required=True,
                    help="CSV for training split (columns: pth[,label],valence,arousal)")
    ap.add_argument("--test_csv",   type=str, required=True,
                    help="CSV for test/validation split (same columns)")
    ap.add_argument("--train_folder", type=str, required=True,
                    help="Folder with training images (paths in train_csv are relative to here)")
    ap.add_argument("--test_folder",  type=str, required=True,
                    help="Folder with test images (paths in test_csv are relative to here)")
                    
    # model / training
    ap.add_argument("--nclasses",  type=int, default=8, choices=[5, 8], help="expression classes")
    ap.add_argument("--use_expr",  action="store_true",
                    help="include expression head training (needs 'label' column)")
    ap.add_argument("--epochs",    type=int, default=40)
    ap.add_argument("--batch",     type=int, default=32)
    ap.add_argument("--size",      type=int, default=256)
    ap.add_argument("--seed",      type=int, default=42)
    ap.add_argument("--lr",        type=float, default=3e-4, help="LR for trainable (last) layers")
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--lambda_expr",  type=float, default=1.0)
    ap.add_argument("--warmup_epochs", type=int, default=0,
                    help="epochs to keep backbone frozen (already frozen by default)") # not currently in use
    ap.add_argument("--unfreeze_backbone_after", type=int, default=0,
                    help="epoch to start unfreezing backbone (>0 enables). Example: 10")
    ap.add_argument("--outdir", type=str, default="runs/emonet_train")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--labelmap_out", type=str, default="runs/label2id.json")
    ap.add_argument("--va_loss_func", type=str, help="Loss function for valence-arousal (continuous)",
                    choices=["ccc_loss", "mse", "mae", "hubert"], default="ccc_loss")
    args = ap.parse_args()
    
    main(args)
