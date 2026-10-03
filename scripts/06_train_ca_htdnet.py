import os
import sys
import json
import time
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.ca_htdnet import CA_HTDNet
from src.models.loss import CostSensitiveFocalLoss
from src.evaluation.metrics import evaluate_classification_performance

os.environ["OMP_NUM_THREADS"] = "1"

def main():
    print("=" * 70)
    print("STAGE 6: TRAINING PROPOSED ARCHITECTURE — CA-HTDNet")
    print("=" * 70)

    os.makedirs("models/proposed", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Target Hardware Execution Device: {device} (Apple M2 Unified Memory)")

    # 1. Load Data
    data = np.load("data/processed/processed_arrays.npz")
    X_train_num = data["X_train_num"]
    y_train = data["y_train"]

    X_val_num = data["X_val_num"]
    y_val = data["y_val"]

    # 2. Load GBDT hints
    print("Loading pre-computed GBDT probability hints...")
    hints_path = "data/processed/gbdt_hints.npz"
    if not os.path.exists(hints_path):
        import subprocess
        subprocess.run([sys.executable, "-c", """
import pickle, numpy as np
data = np.load('data/processed/processed_arrays.npz')
X_train_full = np.hstack([data['X_train_num'], data['X_train_cat']])
X_val_full = np.hstack([data['X_val_num'], data['X_val_cat']])
with open('models/baselines/LightGBM.pkl', 'rb') as f:
    lgb_model = pickle.load(f)
with open('models/baselines/CatBoost.pkl', 'rb') as f:
    cat_model = pickle.load(f)
p_lgb_train = lgb_model.predict_proba(X_train_full)
p_cat_train = cat_model.predict_proba(X_train_full)
gbdt_train = np.hstack([p_lgb_train, p_cat_train]).astype(np.float32)
p_lgb_val = lgb_model.predict_proba(X_val_full)
p_cat_val = cat_model.predict_proba(X_val_full)
gbdt_val = np.hstack([p_lgb_val, p_cat_val]).astype(np.float32)
np.savez_compressed('data/processed/gbdt_hints.npz', gbdt_train=gbdt_train, gbdt_val=gbdt_val)
"""], check=True)

    hints = np.load(hints_path)
    gbdt_train = hints["gbdt_train"]
    gbdt_val = hints["gbdt_val"]

    # 3. Create DataLoaders
    batch_size = 512
    train_ds = TensorDataset(
        torch.tensor(X_train_num, dtype=torch.float32),
        torch.tensor(gbdt_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    val_ds = TensorDataset(
        torch.tensor(X_val_num, dtype=torch.float32),
        torch.tensor(gbdt_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False)

    # 4. Instantiate Model
    num_features = X_train_num.shape[1]
    model = CA_HTDNet(
        num_numerical=num_features,
        num_categories=3,
        d_model=64,
        tcn_channels=[32, 64],
        gru_hidden=64,
        num_classes=2,
        seq_len=8,
        dropout=0.1
    ).to(device)

    # Compute inverse class frequencies for focal loss
    class_counts = np.bincount(y_train)
    weights = len(y_train) / (len(class_counts) * class_counts.astype(float))
    cw_tensor = torch.tensor(weights, dtype=torch.float32).to(device)

    criterion = CostSensitiveFocalLoss(alpha=0.75, gamma=2.0, fn_penalty=2.0, class_weights=cw_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=8, eta_min=1e-5)

    epochs = 8
    best_val_f1 = 0.0
    best_state = None

    print(f"\nStarting CA-HTDNet Training ({epochs} epochs)...")
    t0_train = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for bx_num, b_gbdt, by in train_loader:
            bx_num, b_gbdt, by = bx_num.to(device), b_gbdt.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx_num, b_gbdt)
            loss = criterion(out["logits"], by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(by)
            
        scheduler.step()
        avg_train_loss = train_loss / len(train_ds)

        # Validation
        model.eval()
        val_preds, val_probs = [], []
        with torch.no_grad():
            for bx_num, b_gbdt, by in val_loader:
                bx_num, b_gbdt = bx_num.to(device), b_gbdt.to(device)
                out = model(bx_num, b_gbdt)
                p = torch.softmax(out["calibrated_logits"], dim=-1).cpu().numpy()
                val_probs.append(p)
                val_preds.append(np.argmax(p, axis=1))

        v_probs = np.concatenate(val_probs, axis=0)
        v_preds = np.concatenate(val_preds, axis=0)

        v_metrics = evaluate_classification_performance(y_val, v_preds, v_probs)
        v_f1 = v_metrics["Macro_F1"]
        v_acc = v_metrics["Accuracy"]
        v_rec = v_metrics["Macro_Recall"]

        print(f"Epoch {epoch:2d}/{epochs:2d} | Train Loss: {avg_train_loss:.4f} | Val Acc: {v_acc*100:.2f}% | Val Macro-F1: {v_f1*100:.2f}% | Val Recall: {v_rec*100:.2f}%")

        if v_f1 > best_val_f1:
            best_val_f1 = v_f1
            best_state = model.state_dict().copy()

    total_training_time = time.time() - t0_train
    print(f"\nTraining completed in {total_training_time:.2f}s. Best Val Macro-F1: {best_val_f1*100:.2f}%")

    if best_state is not None:
        model.load_state_dict(best_state)

    # Save model weights
    torch.save(model.state_dict(), "models/proposed/ca_htdnet.pt")
    
    # Save config
    config = {
        "model_architecture": "CA-HTDNet (Crypto-Agile Healthcare Threat Detection Network)",
        "num_numerical": num_features,
        "d_model": 64,
        "tcn_channels": [32, 64],
        "gru_hidden": 64,
        "seq_len": 8,
        "training_time_s": float(total_training_time),
        "best_val_macro_f1": float(best_val_f1)
    }
    with open("models/proposed/config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Final Validation Evaluation
    t_inf_start = time.time()
    model.eval()
    val_preds, val_probs = [], []
    with torch.no_grad():
        for bx_num, b_gbdt, by in val_loader:
            bx_num, b_gbdt = bx_num.to(device), b_gbdt.to(device)
            out = model(bx_num, b_gbdt)
            p = torch.softmax(out["calibrated_logits"], dim=-1).cpu().numpy()
            val_probs.append(p)
            val_preds.append(np.argmax(p, axis=1))

    inf_time = time.time() - t_inf_start
    final_probs = np.concatenate(val_probs, axis=0)
    final_preds = np.concatenate(val_preds, axis=0)

    model_size_kb = os.path.getsize("models/proposed/ca_htdnet.pt") / 1024.0
    param_count = sum(p.numel() for p in model.parameters())

    final_val_metrics = evaluate_classification_performance(
        y_true=y_val,
        y_pred=final_preds,
        y_prob=final_probs,
        training_time=total_training_time,
        inference_time=inf_time,
        model_size_kb=model_size_kb,
        param_count=param_count
    )

    with open("models/proposed/val_metrics.json", "w") as f:
        json.dump(final_val_metrics, f, indent=2)

    print("\n" + "=" * 70)
    print("PROPOSED MODEL CA-HTDNet VALIDATION RESULTS:")
    print("=" * 70)
    print(f"Accuracy:         {final_val_metrics['Accuracy']*100:.2f}%")
    print(f"Macro Precision:  {final_val_metrics['Macro_Precision']*100:.2f}%")
    print(f"Macro Recall:     {final_val_metrics['Macro_Recall']*100:.2f}%")
    print(f"Macro F1:         {final_val_metrics['Macro_F1']*100:.2f}%")
    print(f"ROC-AUC:          {final_val_metrics['ROC_AUC']:.4f}")
    print(f"PR-AUC:           {final_val_metrics['PR_AUC']:.4f}")
    print(f"FPR:              {final_val_metrics['FPR']*100:.3f}%")
    print(f"FNR:              {final_val_metrics['FNR']*100:.3f}%")
    print(f"Inference Latency:{final_val_metrics['Inference_Time_s']*1000/len(y_val):.4f} ms/sample")
    print(f"Throughput:       {final_val_metrics['Throughput_Samples_Sec']:.1f} samples/sec")

if __name__ == "__main__":
    main()
