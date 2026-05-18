import os
import h5py
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import shap
import matplotlib.pyplot as plt
import seaborn as sns

# 커스텀 패키지 함수 임포트
from src.utils import seed_everything, eval_metrics, interpret_confidence, make_optimizer
from src.dataset import H5GeologyDataset, get_img_transform, denormalize
from src.models import GatedResidualFusion, GradCAM

def extract_features_for_xgb(loader, model, device, desc="Extracting"):
    preds_list, gts_list, features_list, labels_list = [], [], [], []
    with torch.no_grad():
        for images, xrfs, y, labels in tqdm(loader, desc=desc):
            images, xrfs = images.to(device), xrfs.to(device)
            pred, _, _, img_feat = model(images, xrfs)
            
            preds_list.append(pred.cpu().numpy())
            gts_list.append(y.numpy())
            
            rgb_mean = images.mean(dim=(2, 3))
            rgb_std  = images.std(dim=(2, 3))
            combined = torch.cat([pred, img_feat, rgb_mean, rgb_std, xrfs], dim=1)
            
            features_list.append(combined.cpu().numpy())
            labels_list.append(labels.numpy())
            
    return np.concatenate(features_list, axis=0), np.concatenate(labels_list, axis=0), np.concatenate(preds_list, axis=0), np.concatenate(gts_list, axis=0)

def main():
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 설정값 구성
    H5_PATH = "/content/drive/MyDrive/Core_data/U1424A_VATT_Final_Labeled_Dataset.h5"
    SAVE_PATH = "/content/drive/MyDrive/Best_Model/best.pth"
    XGB_SAVE_PATH = "/content/drive/MyDrive/Best_Model/xgb_best.joblib"
    
    if not os.path.exists(H5_PATH):
        print(f"[ERROR] Dataset path not found: {H5_PATH}")
        return

    # 대용량 지질 정보 및 고정 레이블 탐색
    with h5py.File(H5_PATH, "r") as f:
        total_samples = len(f["labels"])
        xrf_dim = f["xrf"].shape[1]
        raw_labels = f["labels"][:]
        if len(raw_labels) > 0 and isinstance(raw_labels[0], bytes):
            raw_labels = [l.decode('utf-8') for l in raw_labels]
        else:
            raw_labels = [str(l) for l in raw_labels]
        unique_labels = sorted(list(set(raw_labels)))
        LABEL_MAP = {label: idx for idx, label in enumerate(unique_labels)}
        print(f"[INFO] Classes Count: {len(unique_labels)}")

    # 희귀 샘플 데이터 예외 처리
    label_counts = Counter(raw_labels)
    valid_indices = [i for i, label in enumerate(raw_labels) if label_counts[label] >= 10]
    filtered_idx = np.array(valid_indices)
    filtered_labels = [raw_labels[i] for i in valid_indices]

    # 데이터 계층 구조별 균등 3-way Split
    train_idx, temp_idx, _, temp_labels = train_test_split(filtered_idx, filtered_labels, test_size=0.2, random_state=42, shuffle=True, stratify=filtered_labels)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42, shuffle=True, stratify=temp_labels)

    img_transform = get_img_transform()
    train_ds = H5GeologyDataset(H5_PATH, train_idx, label_map=LABEL_MAP, transform=img_transform)
    val_ds   = H5GeologyDataset(H5_PATH, val_idx,   label_map=LABEL_MAP, transform=img_transform)
    test_ds  = H5GeologyDataset(H5_PATH, test_idx,  label_map=LABEL_MAP, transform=img_transform)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

    # Stage 1: Gated Fusion 기법 기반 물리치 추론 네트워크 학습 가동
    model = GatedResidualFusion(xrf_dim=xrf_dim).to(device)
    criterion_pred = nn.SmoothL1Loss(beta=0.05)
    criterion_base = nn.SmoothL1Loss(beta=0.05)
    
    lr_config = {"LR_XRF": 1e-3, "LR_IMG": 3e-4, "LR_GATE": 3e-4, "LR_ALL": 5e-4, "WEIGHT_DECAY": 1e-2}
    best_r2_mean = -1e9

    print("\n--- [Stage 1] 딥러닝 멀티모달 특성 융합 최적화 가동 ---")
    for epoch in range(50):  # 데모 아카이빙용 최적 루프 50회 설정
        if epoch < 15: stage = "warmup"
        elif epoch < 30: stage = "residual"
        else: stage = "finetune"

        if epoch in [0, 15, 30]:
            optimizer = make_optimizer(model, stage, lr_config)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

        model.train()
        for images, xrfs, y, _ in train_loader:
            images, xrfs, y = images.to(device), xrfs.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred, pred_xrf, _, _ = model(images, xrfs)
            loss = criterion_pred(pred, y) + 0.5 * criterion_base(pred_xrf, y)
            loss.backward()
            optimizer.step()

        # Validation 평가 루틴 생략 및 체크포인트 세이빙
        model.eval()
        preds, gts = [], []
        with torch.no_grad():
            for images, xrfs, y, _ in val_loader:
                images, xrfs = images.to(device), xrfs.to(device)
                pred, _, _, _ = model(images, xrfs)
                preds.append(pred.cpu().numpy())
                gts.append(y.numpy())
        r2s, _, _ = eval_metrics(np.concatenate(preds, 0), np.concatenate(gts, 0))
        r2_mean = np.mean(r2s)
        
        if r2_mean > best_r2_mean:
            best_r2_mean = r2_mean
            os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
            torch.save({"model_state_dict": model.state_dict(), "xrf_dim": xrf_dim, "best_r2_mean": best_r2_mean}, SAVE_PATH)

    print(f"-> Stage 1 완료 (Best R2: {best_r2_mean:.4f})")

    # Stage 2: XGBoost 앙상블 기반 광물 분류기 훈련 시작
    print("\n--- [Stage 2] XGBoost 결합형 고정밀 광물 분류 분석 시작 ---")
    checkpoint = torch.load(SAVE_PATH)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    X_train_xgb, y_train_xgb, _, _ = extract_features_for_xgb(train_loader, model, device, "XGB Train")
    X_val_xgb,   y_val_xgb,   _, _ = extract_features_for_xgb(val_loader,   model, device, "XGB Val")
    X_test_xgb,  y_test_xgb,  preds_p, gts_p = extract_features_for_xgb(test_loader,  model, device, "XGB Test")

    X_train_full = np.concatenate([X_train_xgb, X_val_xgb], axis=0)
    y_train_full = np.concatenate([y_train_xgb, y_val_xgb], axis=0)

    le = LabelEncoder()
    y_train_full_enc = le.fit_transform(y_train_full)
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train_full_enc)

    xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, n_jobs=-1)
    xgb_model.fit(X_train_full, y_train_full_enc, sample_weight=sample_weights)

    os.makedirs(os.path.dirname(XGB_SAVE_PATH), exist_ok=True)
    joblib.dump({'model': xgb_model, 'le': le}, XGB_SAVE_PATH)
    
    test_preds_enc = xgb_model.predict(X_test_xgb)
    test_preds_xgb = le.inverse_transform(test_preds_enc)
    print(f"[RESULT] 최적 검증셋 기준 XGBoost Classification 최종 정확도: {accuracy_score(y_test_xgb, test_preds_xgb):.4f}")

    # Stage 3: 사후 시각화 및 변수 중요도(SHAP) 분석 플롯 추출
    print("\n--- [Stage 3] 설명 가능한 AI 모델 근거 데이터 플로팅 완료 ---")
    feature_names = ["Pred_PWV", "Pred_Density", "Pred_MS"] + [f"Img_Latent_{i}" for i in range(8)] + ["R_mean", "G_mean", "B_mean", "R_std", "G_std", "B_std"] + [f"GT_XRF_{i}" for i in range(xrf_dim)]
    
    importances = xgb_model.feature_importances_
    df_imp = pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values(by="Importance", ascending=False).head(5)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=df_imp, palette="viridis")
    plt.title("XGBoost Multimodal Feature Importance (Top 5)")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    print("-> 중요도 플롯 저장 완료: 'feature_importance.png'")

    # Clean Up
    train_ds.close(); val_ds.close(); test_ds.close()
    print("✅ 모든 실험 및 의존성 아카이빙 정상 종료.")

if __name__ == "__main__":
    main()
