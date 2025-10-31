import argparse
from pathlib import Path
import pandas as pd
import time
import torch
from torch.utils.data import DataLoader

from emonet.models import EmoNet
from emonet.data import EmoNetCSV

import numpy as np
from sklearn.metrics import confusion_matrix


#region LABELS

# Canonical class orders used by EmoNet expression head.
CANONICAL_8 = ["neutral","happy","sad","surprise","fear","disgust","anger","contempt"]
CANONICAL_5 = ["neutral","happy","sad","surprise","fear"] 

def build_fixed_label_map(nclasses: int):
    if nclasses == 8:   order = CANONICAL_8
    elif nclasses == 5: order = CANONICAL_5
    else:               raise ValueError("nclasses must be 5 or 8")
    return {name: i for i, name in enumerate(order)}

def get_dataset_label2id(csv: str) -> dict|None:
    df_train = pd.read_csv(csv)
    label2id = None
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


#region EVALUATION

def ccc(x, y, eps=1e-8):
    """
    Concordance Correlation Coefficient ([-1, 1]).
    Higher is better.
    """
    x = x.float(); y = y.float()
    xm, ym = x.mean(), y.mean()
    xv, yv = x.var(unbiased=False), y.var(unbiased=False)
    cov = ((x - xm) * (y - ym)).mean()
    return (2 * cov) / (xv + yv + (xm - ym) ** 2 + eps)

def ccc_loss(x, y): return 1.0 - ccc(x, y)

def rmse(x, y): return torch.sqrt(torch.mean((x - y) ** 2))
def mae(x, y):  return torch.mean(torch.abs(x - y))



# Evaluation
def evaluate_emonet(model: torch.nn.Module, loader: DataLoader, device: str, use_expr: bool=True) -> dict[str, float]:
    """
    Run validation over a loader and compute VA metrics (CCC/RMSE/MAE),
    plus expression accuracy if enabled.
    """
    model.eval()
    v_pred, v_true, a_pred, a_true = [], [], [], []
    # expr_correct, expr_total = 0, 0
    expr_preds, expr_trues = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            vp = out["valence"].view(-1).cpu()
            ap = out["arousal"].view(-1).cpu()
            v_pred.append(vp); a_pred.append(ap)
            v_true.append(y["valence"]); a_true.append(y["arousal"])
            if use_expr and "expression" in out and "expr" in y:
                preds = out["expression"].argmax(1).cpu()
                # expr_correct += (pred == y["expr"]).sum().item()
                # expr_total += pred.numel()
                expr_preds.extend(preds)
                expr_trues.extend(y["expr"].cpu())
                
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

    # expression
    cm = confusion_matrix(expr_trues, expr_preds)
    total_acc = np.trace(cm) / np.sum(cm)
    class_acc = np.diag(cm) / np.sum(cm, axis=1) # per class accuracy (axis=0 for precision)
    
    if use_expr:
        metrics["expr_acc_total"] = total_acc
        for i, acc in enumerate(class_acc):
            metrics[f"expr_acc_{CANONICAL_8[i]}"] = acc # HUOM: Currently only works for 8 classes
    return metrics


#region MAIN

def main(args: argparse.Namespace):

    device = 'cuda'
    if not torch.cuda.is_available():
        print('[MAIN] WARNING: cuda not available, using cpu')
        device = 'cpu'
    
    label2id = get_dataset_label2id(args.csv)


    # Datasets & Loaders
    dataset  = EmoNetCSV(args.csv,  args.dataset_root,  size=256,
                         use_expr=True, label2id=label2id, augment=False)

    data_loader  = DataLoader(dataset,  batch_size=args.batch, shuffle=False,
                              num_workers=4, pin_memory=True)


    # Model & Params
    pretrained_params = Path(args.pretrained_params)
    if not pretrained_params.exists():
        raise FileNotFoundError(f"no pretrained parameters file found: {pretrained_params}")
    
    print(f"Loading parameters: {pretrained_params}")
    state = torch.load(str(pretrained_params), map_location="cpu")
    state = { k.replace("module.", ""): v for k, v in state.items() }

    model = EmoNet(n_expression=args.nclasses).to(device)
    model.load_state_dict(state, strict=False)


    # Run evaluation
    print("[MAIN] running evaluation ...")
    t0 = time.time()
    vl = evaluate_emonet(model, data_loader, device, use_expr=True)
    print("[MAIN] done. took {:.1f} sec".format(time.time()-t0))

    # format results
    keys_str = ''
    values_str = ''
    for i, (k, v) in enumerate(vl.items()):
        keys_str += k
        values_str += f'{v:.4f}'
        if i < len(vl)-1:
            keys_str += ','; values_str += ','
    
    print('\n  EVALUATION RESULTS:')
    print(keys_str)
    print(values_str)



#region CLI
if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    
    # data paths
    ap.add_argument("--pretrained_params", type=str, help="Path to pretrained params/weights (.pth file)", required=True)
    ap.add_argument("--dataset_root",  type=str, required=True,
                    help="path to dataset root")
    ap.add_argument("--csv",  type=str, required=True,
                    help="CSV for training split (columns: pth[,label],valence,arousal)")
    
    # model / training
    ap.add_argument("--nclasses",  type=int, default=8, choices=[5, 8], help="expression classes")
    ap.add_argument("--batch",     type=int, default=32)
    
    args = ap.parse_args()
    
    main(args)
