# 待辦事項

## 模型優化（剩餘兩天）

- [ ] **A. LightGBM**（今天）
  - 建 `level2_lgbm.ipynb`
  - 比較 AUC vs GBM 基準 0.7051
  - 若有提升 → 上傳排行榜

- [ ] **B. Optuna 超參數調優**（今晚 overnight）
  - 在 LightGBM 或 GBM 上跑 5-fold CV AUC 最大化
  - 搜尋空間：n_estimators, max_depth, learning_rate, min_child_samples, reg_alpha, reg_lambda
  - 明早看結果

- [ ] **C. Ensemble**（明天，若 A/B 都有提升）
  - 平均 GBM + LightGBM（+ 最佳 Optuna 模型）的 P_gave
  - 預期 +0.003–0.01 AUC

## 簡報準備

- [ ] 整理比賽策略與實驗結果
- [ ] 製作簡報（若進入前十名）

## 跳出框架的替代方案

- [ ] **X-Learner**（進行中）— `level2_xlearner.ipynb`
  - 解決 T-Learner 的 Contact=1 樣本少（10K）問題
  - 用 M0 補齊 Contact=1 的反事實，訓練 CATE 估計器
  - 加入 Propensity Score 加權

- [ ] **Tweedie/Zero-Inflated 回歸**（選做）
  - 直接建模 AmtThisYear 的零膨脹分布（82% 是 0）
  - 用 TweedieRegressor(power=1.5) 取代目前的 P×E 兩段式

- [ ] **Cost-Sensitive Learning**（選做）
  - 直接以「聯繫是否划算」（AmtThisYear > $5 且 Contact=1）為 target
  - 繞過 P×E 分解，讓模型直接學習業務目標

## 已完成

- [x] Level 1：EV 框架（GBM），AUC 0.6854，Surplus $8,246,195
- [x] Level 2 Single Model：加入 Contact feature，AUC 0.7051，Surplus $8,688,900
- [x] Level 2 T-Learner：Holdout AUC(M1)=0.5914，Surplus $8,674,820（樣本不足過擬合）
- [x] 特徵工程：AUC 無提升（GBM 已內化這些組合）
- [x] GBM Optuna：CV AUC 0.7084，Surplus **$8,707,735** ✅ 目前最佳
- [x] LightGBM Optuna：Surplus $8,702,410
- [x] XGBoost Optuna：Surplus $8,706,990
- [x] 三模型 Optuna 後皆收斂至 ~$8.70M，代表資料集可預測性上限已接近
