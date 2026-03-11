# Level 1 實驗紀錄

資料：`R1_TRAIN`（99,951 有效樣本）→ 評分 `R1_SCOREDATA`（100,000 筆）

---

## 資料基本資訊

| 項目 | 數值 |
|------|------|
| 有效樣本數 | 99,951（49 筆因 NaN 被丟棄） |
| 捐款率 | 18.05% |
| 捐款者平均金額 | $65.21 |
| 捐款金額中位數 | $25 |
| 捐款金額最大值 | $10,000 |
| 特徵數 | 19（含 one-hot） |

---

## 模型架構：EV 框架

```
EV_i = P(GaveThisYear=1) × E(AmtThisYear | gave=1)
聯繫條件：EV_i > 邊際成本
```

成本結構：前 60,000 人 $5/人，超過 $25/人

---

## 模型一：分類模型 P(GaveThisYear=1)

- 演算法：GradientBoostingClassifier（200 trees, depth 4, lr 0.05, subsample 0.8）
- 訓練/驗證：80/20 stratified split

| 指標 | 數值 |
|------|------|
| AUC | 0.6854 |
| Recall（捐款者，threshold=0.5） | 4%（門檻效果差，但 ranking 有效） |

**機率校準（驗證集）：**

| 預測概率區間 | 實際捐款率 |
|------------|----------|
| 0.0–0.1 | 7.8% |
| 0.1–0.2 | 14.2% |
| 0.2–0.3 | 24.5% |
| 0.3–0.5 | 37.8% |
| 0.5–0.7 | 56.1% |
| 0.7–1.0 | 75.0% |

方向正確，校準合理。不需要下採樣（EV 框架依賴排序品質而非 0.5 切點）。

---

## 模型二：回歸模型 E(AmtThisYear | gave=1)

- 只在捐款者上訓練（train: 14,433 人）
- 演算法：GradientBoostingRegressor（300 trees, depth 4, lr 0.05）
- 訓練目標：`log1p(AmtThisYear)`

| 指標 | 數值 | 說明 |
|------|------|------|
| R²（log space） | **0.0179** | 幾乎學不到東西 |
| R²（original space） | -0.0165 | 被極端值($10K)炸掉 |
| MAE | $47.42 | |
| RMSE | $217.02 | |

**結論：** 金額難以預測，R² 近 0。但對 EV 排序仍有 Spearman 貢獻（見下方比較）。

---

## 策略比較（驗證集）

| 策略 | Spearman(EV, 實際金額) | 聯繫人數 | Surplus |
|------|----------------------|---------|---------|
| 回歸模型 EV | **0.2331** | 9,704 | **$126,105** |
| AmtLastYear 代理 | 0.1713 | 17,153 | $123,640 |
| 真實最佳（事後） | — | 13,965 | $140,270 |
| 全部聯繫 | — | 19,991 | $128,550 |

**AmtLastYear 代理方法：** `E_amount = AmtLastYear if AmtLastYear > 0 else $65.21`
→ 回歸模型 Spearman 更高，Surplus 更高，不採用 AmtLastYear 代理。

---

## 聯繫名單策略

### 策略 A：EV > 邊際成本（保守）

```python
contact = score_df[score_df['EV'] > score_df['marginal_cost']]
```

| 項目 | 數值 |
|------|------|
| 聯繫人數 | 49,700（49.7%） |
| 預估成本 | $248,500 |
| EV 門檻 | $5.00 |
| 輸出 | `output/level1_contact_list.csv` |

### 策略 B：Top N%（驗證集最佳比例外推）

驗證集最佳切點 13,965 / 19,991 = **69.85%**，套到 SCOREDATA（100K 人）：

```python
TOP_FRAC = 0.6985
contact_topn = score_df.head(int(TOP_FRAC * len(score_df)))
```

| 項目 | 數值 |
|------|------|
| 聯繫人數 | 69,851（69.9%） |
| 預估成本 | $546,275 |
| 輸出 | `output/level1_topn_contact_list.csv` |

### 策略比較（含實際排行榜結果）

| | 策略 A | 策略 B |
|-|--------|--------|
| 人數 | 49,700 | 69,851 |
| 成本 | $248,500 | $546,275 |
| 排行榜結果 | **略高** ✅ | 略低 |
| 結論 | 採用 | 捨棄 |

**觀察：** 策略 B 多聯繫 20,151 人，但成本多 $297,775，邊際貢獻為負。
代表 EV 排序第 49,701–69,851 名的人帶來的捐款不足以覆蓋 $5–$25 的聯繫成本。
驗證集的「最佳比例 70%」在 SCOREDATA 上過於樂觀，不適用。

---

## 待辦 / 可改進方向

- [x] 上傳兩種策略到排行榜 → 策略 A（EV > 門檻）勝出
- [ ] 改善分類模型（AUC 0.6854 仍有提升空間）：嘗試 LightGBM、特徵工程
- [ ] 特徵工程：AvgGift = TotalGift / Frequency、MaxGift / TotalGift 比率
- [ ] Level 2：利用 Contact/NoContact uplift 識別 Persuadables 與 Sleeping Dogs
