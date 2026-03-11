# NYU × SAS Cortex Analytics Challenge — 完整訓練報告

**隊伍：** Prophet Margins
**最終成果：** Operating Surplus $8,707,735（排行榜第一）
**最佳提交：** Level 2 + Optuna 超參數調優

---

## 一、比賽目標與評分機制

**目標：** 從 100,000 名潛在捐款者中，選出最佳聯繫名單，最大化 Operating Surplus。

$$Operating\ Surplus = \sum(\text{被聯繫者的實際捐款}) - \sum(\text{聯繫成本})$$

**成本結構（累進階梯制）：**
- 前 60,000 人：$5 / 人
- 超過 60,000 人：$25 / 人

**輸出格式：** 僅含 `ID` 欄位的 CSV，上傳至排行榜。排行榜後台持有實際捐款資料，根據提交的 ID list 計算真實 Surplus。

---

## 二、資料集說明

| 資料集 | 筆數 | 特有欄位 | 用途 |
|--------|------|---------|------|
| `R1_TRAIN` | 100,000 | GaveThisYear, AmtThisYear | Level 1 訓練 |
| `R1_SCOREDATA` | 100,000 | 無 Target | Level 1 評分 |
| `R2_TRAIN` | 100,000 | Contact, GaveThisYear, AmtThisYear | Level 2 訓練 |
| `R2_CONTACT_SCOREDATA` | 100,000 | Contact=1，無 Target | Level 2 評分（聯繫情境） |
| `R2_NOCONTACT_SCOREDATA` | 100,000 | Contact=0，無 Target | Level 2 評分（不聯繫情境） |

**時間位移：** TRAIN 的 ThisYear 對應 SCOREDATA 的 LastYear，SCOREDATA 無 Target 欄位，代表「未來一年的捐款結果」由排行榜後台持有。

**R2 資料的 Contact 分佈：**
- Contact=0（未聯繫）：89,886 人，捐款率 18.1%
- Contact=1（已聯繫）：10,114 人，捐款率 39.9%
- **聯繫效應：捐款率提升 +21.8%**，但捐款金額差異極小（$67 vs $70）

---

## 三、核心策略框架（EV 模型）

所有方法都基於期望值（Expected Value）框架：

$$EV_i = P(\text{捐款}_i) \times E(\text{金額}_i \mid \text{捐款})$$

**聯繫決策準則：**
- 若 $EV_i >$ 邊際成本 → 聯繫
- 若 $EV_i \leq$ 邊際成本 → 不聯繫

Level 2 進一步計算 Uplift（聯繫帶來的增量效益）：

$$Uplift_i = EV_i(Contact=1) - EV_i(Contact=0)$$

依此將所有人分為四象限：

| | 不聯繫捐款率高 | 不聯繫捐款率低 |
|---|---|---|
| **聯繫後效益高** | Sure Things（省成本，不聯繫） | **Persuadables（目標）** |
| **聯繫後效益低** | **Sleeping Dogs（避開！）** | Lost Causes（不聯繫） |

---

## 四、Level 1：基礎 EV 建模

### 方法

- **訓練資料：** R1_TRAIN（99,951 有效樣本，19 個特徵）
- **評分資料：** R1_SCOREDATA
- **模型架構：**
  - 分類模型：GBM → P(GaveThisYear=1)
  - 回歸模型：GBM（log 目標）→ E(AmtThisYear | gave=1)，僅在捐款者上訓練
  - 最終 EV = P × E

### 資料前處理

- BOM 清理（`encoding='utf-8-sig'`）、去除引號空白
- 目標欄位含 NaN 的 49 筆資料丟棄（剩 99,951 筆）
- Education、City 做 One-Hot Encoding
- 80/20 Stratified Split 做驗證

### 模型表現

| 指標 | 數值 |
|------|------|
| 分類 AUC（驗證集） | 0.6854 |
| 回歸 R²（log space） | 0.0179 |
| 回歸 R²（original space） | -0.0165（被極端值拉低） |

**機率校準驗證（P 越高實際率越高，排序有意義）：**

| 預測概率區間 | 實際捐款率 |
|------------|----------|
| 0.0 – 0.1 | 7.8% |
| 0.1 – 0.2 | 14.2% |
| 0.2 – 0.3 | 24.5% |
| 0.3 – 0.5 | 37.8% |
| 0.5 – 0.7 | 56.1% |
| 0.7 – 1.0 | 75.0% |

**回歸模型說明：** R²（log space）= 0.0179，模型學到的訊號非常微弱，主要原因是捐款金額右偏嚴重（中位數 $25，但最大值 $10,000），少數極端值主導了殘差。驗證後確認回歸模型仍比使用固定均值（AmtLastYear 代理法，Spearman = 0.1713）更能排序（Spearman = 0.2331），故保留。

### 聯繫策略比較

| 策略 | 聯繫人數 | 預估成本 | 排行榜 Surplus |
|------|---------|---------|--------------|
| EV > 邊際成本 | 49,700 | $248,500 | **$8,246,195** ✅ |
| Top N%（驗證最佳比例外推）| 69,851 | $545,525 | $8,222,640 |

**結論：** EV > 邊際成本策略勝出。Top N% 策略多聯繫 20,151 人，但成本暴增 $297K，邊際貢獻為負。

---

## 五、Level 2：Uplift 建模（Single Model）

### 核心改進

**只訓練一個模型，但 `Contact` 保留為 feature**（0 或 1）。評分時，對每個人預測兩次：
- 強制設 `Contact=1` → 得到 EV_contact
- 強制設 `Contact=0` → 得到 EV_nocontact

```
EV_contact   = P(捐 | Contact=1) × E(金額 | 捐, Contact=1)
EV_nocontact = P(捐 | Contact=0) × E(金額 | 捐, Contact=0)
Uplift       = EV_contact - EV_nocontact
```

**「Single Model」的含義：** EV 計算本身永遠需要兩個模型——分類器（預測捐款機率）加回歸器（預測捐款金額）。「Single」指的是每種模型只訓練一個，Contact 只是其中一個 feature，不是像 T-Learner 那樣分兩組各自訓練兩個分類器和兩個回歸器。

| 版本 | 分類器 | 回歸器 | 合計 |
|------|--------|--------|------|
| Level 2 Single Model | 1 個（含 Contact feature）| 1 個（含 Contact feature）| **2 個** |
| T-Learner | 2 個（M0 + M1）| 2 個（R0 + R1）| 4 個 |

### 四象限分類邏輯

依 **Uplift** 和 **EV_nocontact** 兩個維度，將每個人分到四個象限：

```python
if uplift > marginal_cost:                        → Persuadable   （聯繫）
elif EV_nocontact > $5 and uplift >= 0:           → Sure Thing    （不聯繫）
elif uplift < 0:                                  → Sleeping Dog  （絕對不聯繫）
else:                                             → Lost Cause    （不聯繫）
```

| | EV_nocontact 高（> $5） | EV_nocontact 低（≤ $5） |
|---|---|---|
| **Uplift > 成本** | Persuadable | Persuadable |
| **Uplift 0 ～ 成本** | **Sure Thing** | Lost Cause |
| **Uplift < 0** | Sleeping Dog | Sleeping Dog |

**具體範例：**

| 人 | EV_contact | EV_nocontact | Uplift | 分類 | 原因 |
|----|-----------|-------------|--------|------|------|
| A | $30 | $8 | $22 | Persuadable | Uplift $22 > 成本 $5，聯繫划算 |
| B | $25 | $20 | $5 | Sure Thing | EV_nocontact 高，不聯繫也會捐，省下成本 |
| C | $10 | $18 | -$8 | Sleeping Dog | 聯繫後捐款反而少 $8，絕對不聯繫 |
| D | $6 | $2 | $4 | Lost Cause | Uplift $4 < 成本 $5，不值得聯繫 |

**最終只聯繫 Persuadables。** Sure Things 讓他們自己捐（省 $5/人），Sleeping Dogs 主動排除，Lost Causes 忽略。

### 方法

- **訓練資料：** R2_TRAIN（100,000 筆，20 個特徵，含 Contact）
- **評分資料：** R2_CONTACT_SCOREDATA + R2_NOCONTACT_SCOREDATA 各自評分後相減
- **模型架構：** GradientBoostingClassifier + GradientBoostingRegressor，Contact 保留為 feature

**關於標準化（StandardScaler）：** 程式碼中有使用 StandardScaler，但對樹狀模型（GBM、XGBoost、LightGBM、Random Forest）來說**標準化不影響結果**——樹的分割依據是特徵值的相對大小（排序），與尺度無關。標準化只對線性模型、SVM、KNN、神經網絡等有效。程式碼中保留 StandardScaler 是無害但多餘的步驟。

### 模型表現

| 指標 | 數值 |
|------|------|
| 分類 AUC（驗證集） | **0.7051**（比 L1 提升 +0.0197）|
| Contact 特徵重要度 | **20.86%（排名第一）** |
| 策略效率（驗證集）| 70.3% |

**Top 5 特徵重要度：**

| 特徵 | 重要度 | 說明 |
|------|--------|------|
| Contact | 20.86% | 是否被聯繫，模型學到聯繫效應 |
| NbActivities | 17.24% | 活動參與次數 |
| SeniorList | 10.98% | 資深名單 |
| Referrals | 10.34% | 推薦人數 |
| Frequency | 8.74% | 歷史捐款次數 |

### SCOREDATA 四象限分布

| 象限 | 人數 | 處置 |
|------|------|------|
| Persuadables | 53,277 | ✅ 聯繫 |
| Sure Things | 18,358 | 省成本，不聯繫（他們本來就會捐） |
| Sleeping Dogs | 7,523 | 絕對不聯繫（聯繫後捐款反而減少） |
| Lost Causes | 20,842 | 不聯繫 |

**排行榜結果：$8,688,900**（比 Level 1 多 $442,705，+5.4%）

---

## 六、T-Learner（實驗，未採用）

### 方法

T-Learner 的思路是：分別對 Contact=0 和 Contact=1 的人各自訓練**兩個獨立模型**，而不是讓同一個模型學習 Contact 的效應。

- M0：在 Contact=0（89,886 人）上訓練，預測「不聯繫時的捐款行為」
- M1：在 Contact=1（10,114 人）上訓練，預測「聯繫後的捐款行為」

### 問題

M1 的訓練樣本（含 20% 驗證集隔離後剩 8,128 人）不足，導致嚴重過擬合：

| 指標 | 數值 |
|------|------|
| M1 Train AUC | 0.8302（虛高，過擬合）|
| M1 Holdout AUC | **0.5914**（幾乎等於亂猜）|
| R1（回歸）Holdout R² | -0.0548（負值）|

### 有趣發現

T-Learner 的 M0 和 M1 學到了**完全不同的特徵邏輯**：

| 指標 | M0（不聯繫） | M1（有聯繫） |
|------|------------|------------|
| 第一重要特徵 | NbActivities（0.338）| Age（0.205）|
| 邏輯 | 活動參與度 = 自發捐款意願 | 人口特徵 = 對聯繫的反應傾向 |

這印證了一個直覺：**沒有被聯繫時，捐款者是靠活動參與度自發驅動；被聯繫後，年齡、性別、薪資等人口特徵才是決定是否回應的關鍵。**

**排行榜結果：$8,674,820**（比 Level 2 Single Model 低 $14,080）

---

## 七、特徵工程（實驗，無改善）

新增 7 個衍生特徵：

| 特徵 | 公式 | 邏輯 |
|------|------|------|
| AvgGift | TotalGift / Frequency | 平均單次捐款 |
| GiftTrend | AmtLastYear − AvgGift | 捐款趨勢（正 = 上升）|
| DonationRate | Frequency / Seniority | 年均捐款率 |
| GiftRange | MaxGift − MinGift | 捐款波動幅度 |
| RecentActivity | GaveLastYear × AmtLastYear | 近期活躍度 |
| log_TotalGift | log1p(TotalGift) | 壓縮右偏 |
| log_Salary | log1p(Salary) | 壓縮右偏 |

**結果：AUC 維持 0.7051，無任何提升。**

原因：GBM 本身可以學習非線性組合，這些衍生特徵的資訊已隱含在原始特徵中（TotalGift、Frequency、Seniority 都是原始欄位），手動計算的比率未帶來新資訊。

---

## 八、LightGBM（實驗，未採用）

### 測試動機

LightGBM 使用 leaf-wise 生長策略，通常比 sklearn GBM 的 level-wise 更強，加上 `scale_pos_weight` 可以處理 18:82 的類別不平衡。

### 結果

| 指標 | GBM | LightGBM |
|------|-----|---------|
| Holdout AUC | **0.7051** | 0.6966 |
| 5-fold CV AUC | — | 0.6977 ± 0.0036 |
| 策略效率 | **70.3%** | 64.0% |

**Contact 特徵在 LightGBM 中完全消失（Top 10 以外）**，根本原因是 LightGBM 預設使用 split count 量測重要度，Contact 是 binary feature，分割次數少，但每次分割的 gain 很大。加上 `scale_pos_weight` 把模型焦點拉向 Salary、Age 等連續人口特徵，Contact 的效應被稀釋。

**結論：** 不採用 LightGBM，GBM 對此問題更適合。

---

## 九、Optuna 超參數調優（最終最佳方法）

### 方法

使用 Optuna（TPE 採樣）對 GBM 做自動超參數搜尋，100 trials × 5-fold CV AUC 最大化，耗時約 57 分鐘。

**搜尋空間：**

| 超參數 | 範圍 |
|--------|------|
| n_estimators | 100 – 600 |
| max_depth | 2 – 8 |
| learning_rate | 0.01 – 0.3（log scale）|
| subsample | 0.5 – 1.0 |
| min_samples_leaf | 5 – 100 |
| max_features | 0.4 – 1.0 |

**最佳超參數：**

| 參數 | 預設值 | Optuna 最佳 |
|------|--------|------------|
| n_estimators | 200 | **375** |
| max_depth | 4 | **6** |
| learning_rate | 0.05 | **0.0129** |
| subsample | 0.8 | **0.604** |
| min_samples_leaf | 1 | **45** |
| max_features | 1.0 | **0.920** |

### 結果

| 指標 | GBM 基準 | Optuna 最佳 | 提升 |
|------|---------|------------|------|
| 5-fold CV AUC | 0.7051 | **0.7084** | +0.0033 |
| Holdout AUC | 0.7051 | **0.7057** | +0.0006 |
| R²（log，回歸）| 0.0155 | **0.0250** | +0.0095 |
| 聯繫人數 | 53,277 | 52,243 | -1,034 |
| 預估成本 | $266,385 | **$261,215** | -$5,170 |
| **排行榜 Surplus** | $8,688,900 | **$8,707,735** | **+$18,835** |

---

## 十、X-Learner（最終最佳方法）

### 動機：解決 T-Learner 的根本問題

T-Learner 的 M1（Contact=1 群）只有 8,128 筆訓練資料，Holdout AUC 僅 0.5914，幾乎等於亂猜。X-Learner 的解法是**借用 M0（89K 筆，穩定）來幫助 M1 的估計**。

### 演算法（四步驟）

**Step 1：訓練基礎模型**（與 T-Learner 相同）
- M0 在 Contact=0 群訓練 → 估計「不聯繫時的基準捐款率」
- M1 在 Contact=1 群訓練 → 估計「聯繫後的捐款率」

**Step 2：計算偽目標（Pseudo Outcome）** — 核心創新

```
對 Contact=1 的人：D1 = 實際捐款(Y1) − M0(x)
  → D1 > 0：聯繫讓此人「超預期」捐款（Persuadable 訊號）
  → D1 < 0：聯繫後反而低於預期（Sleeping Dog 訊號）

對 Contact=0 的人：D0 = M1(x) − 實際捐款(Y0)
  → D0 > 0：如果聯繫他，預期比現在多貢獻
```

M0 是用 89K 穩定資料訓練的，所以用它估計 Contact=1 群的反事實（D1）不受 8K 樣本限制影響。

**Step 3：訓練 CATE 估計器**
- τ1 在 Contact=1 群訓練，target = D1（個別化治療效應）
- τ0 在 Contact=0 群訓練，target = D0

τ1 雖然仍在 8K 上訓練，但 target（D1）已由穩定的 M0 補強，不再直接預測 0/1 的稀疏標籤。

**Step 4：Propensity Score 加權合併**

```
g(x) = P(Contact=1 | x)   ← 傾向評分模型
CATE = g(x) × τ0(x) + (1−g(x)) × τ1(x)
```

若 g(x) 高（此人本來容易被聯繫）→ 用 τ0（control 群估計）更可靠；若 g(x) 低 → 用 τ1（treatment 群估計）更可靠。

### 模型配置

| 模型 | 訓練資料 | 目標 |
|------|---------|------|
| M0 | Contact=0（71,872 人）| P(gave \| no contact) |
| M1 | Contact=1（8,128 人）| P(gave \| contact) |
| τ0 | Contact=0，target=D0 | CATE 估計（control 群）|
| τ1 | Contact=1，target=D1 | CATE 估計（treatment 群）|
| Propensity | 全量（80,000 人）| P(Contact=1 \| x) |

超參數使用 GBM Optuna 搜尋結果（n_estimators=375, max_depth=6, lr=0.0129, subsample=0.604, min_samples_leaf=45）。

### 關鍵指標

| 指標 | 數值 | 說明 |
|------|------|------|
| τ1 Train R² | 0.3139 | CATE 估計器有效學習個別化效應 |
| τ0 Train R² | 0.2450 | |
| Propensity AUC | 0.7210 | Contact 非隨機，PS 加權有意義 |
| 驗證效率 | **84.9%** | 顯著優於 Single Model（70.3%）|
| CATE > 0（Persuadables）| 84,445 人 | |
| Sleeping Dogs | 15,555 人 | |

### 人均金額個人化實驗（未採用）

嘗試將 CATE（機率單位）乘以個人化金額（AmtLastYear 或 AvgGift）而非固定均值 $65.21。

結果：驗證效率從 84.9% 降至 66.4%，聯繫人數從 60K 降至 49K。

原因：AmtLastYear 有極端值（max $10,000），高歷史金額者的 EV 被過度放大，破壞了 CATE 的排序。固定均值讓 CATE 保持純排序功能，更穩定。

### SCOREDATA 四象限分布

| 象限 | 人數 | 處置 |
|------|------|------|
| Persuadable | 60,000 | ✅ 聯繫（剛好在成本跳漲門檻）|
| Sleeping Dog | 15,555 | 絕對不聯繫 |
| Lost Cause | 24,445 | 不聯繫 |

60,000 人恰好落在成本階梯的跳漲點（前 60K 每人 $5，第 60,001 人起每人 $25），模型自然找到了最優成本邊界。

**排行榜結果：$8,718,670（目前最佳）**

---

## 十一、完整結果總表（排行榜順序）

| 排名 | 方法 | Surplus | 成本 | 聯繫人數 |
|------|------|---------|------|---------|
| **1** | **X-Learner** | **$8,718,670** | $299,860 | 59,972 |
| 2 | GBM Optuna | $8,707,735 | $261,100 | 52,220 |
| 3 | XGBoost Optuna | $8,706,990 | $268,850 | 53,770 |
| 4 | LightGBM Optuna | $8,702,410 | $265,420 | 53,084 |
| 5 | Level 2 Single Model | $8,688,900 | $266,270 | 53,254 |
| 6 | Level 2 T-Learner | $8,674,820 | $254,670 | 50,934 |
| 7 | Level 1 EV > 門檻 | $8,246,195 | $248,375 | 49,675 |
| 8 | Level 1 Top N% | $8,222,640 | $545,525 | 69,821 |
| — | Baseline（不聯繫） | $7,602,655 | $0 | 0 |

**Baseline 說明：** $7,602,655 是完全不做任何聯繫時，自發捐款的總金額。X-Learner 策略在此基礎上額外創造了 **+$1,116,015** 的捐款收益，扣除 $299,860 的聯繫成本，淨增益 **$816,155**。

---

## 十二、主要學習與結論

### 1. Uplift 建模是關鍵

從 Level 1（不考慮 Contact）升級到 Level 2（引入 Uplift）帶來了 **+$442,705（+5.4%）** 的改善。識別並排除 Sure Things（18,358 人，不聯繫也會捐）和 Sleeping Dogs（7,523 人，聯繫後反而少捐）是主要貢獻來源。

### 2. 模型複雜度不是萬靈丹

| 嘗試 | 期望 | 實際 |
|------|------|------|
| 特徵工程 | 提升 AUC | 無效（GBM 已內化組合）|
| T-Learner | 更精準的個別化 Uplift | 樣本不足過擬合（AUC 0.59）|
| LightGBM | 比 GBM 更強 | AUC 反而下降 |
| Optuna | 找到更好的超參數 | +$18,835，有效但幅度小 |
| **X-Learner** | 解決 T-Learner 的樣本不足問題 | **驗證效率 84.9%，+$10,935** ✅ |

**回歸模型（預測捐款金額）始終是瓶頸**：R²（log space）最高只到 0.025，金額極端值導致模型難以學習。分類模型（P(捐款)）的排序能力才是真正的驅動力。

### 3. 資料本身的上限

經過多輪實驗，AUC 始終在 0.70–0.71 之間，代表這個資料集的可預測性有其上限。特徵（人口結構 + 歷史捐款行為）能解釋的變異量有限，剩餘的捐款行為是本質上難以預測的隨機性。

---

## 十三、程式碼結構

| 檔案 | 說明 |
|------|------|
| `level1.ipynb` | Level 1 完整 pipeline |
| `level2.ipynb` | Level 2 Single Model |
| `level2_tlearner.ipynb` | T-Learner 實驗 |
| `level2_fe.ipynb` | 特徵工程實驗 |
| `level2_lgbm.ipynb` | LightGBM 實驗 |
| `level2_optuna.ipynb` | GBM Optuna 調優 |
| `level2_lgbm_optuna.ipynb` | LightGBM Optuna 調優 |
| `level2_xgb_optuna.ipynb` | XGBoost Optuna 調優 |
| `level2_xlearner.ipynb` | **X-Learner（最終最佳，$8,718,670）** |
| `doc/note.md` | 比賽規則與資料說明 |
| `doc/training_report.md` | 本文件 |
| `output/level2_xlearner_contact_list.csv` | **最終提交名單（60,000 人）** |
