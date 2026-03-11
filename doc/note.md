# NYU x SAS Cortex 募款挑戰賽 — 重點摘要

## 比賽目標

**最大化營運盈餘 (Operating Surplus)**

$$Operating\ Surplus = \sum(\text{被聯繫者的實際捐款}) - \sum(\text{聯繫成本})$$

不是聯繫越多人越好，而是要精準篩選「聯繫後淨收益為正」的會員。

---

## 成本結構 (Cost Schedule)

> ⚠️ 成本表依遊戲設定不同而異，以下為主要文件 (p.6) 的版本，請以實際遊戲內為準。

| 聯繫人數區間 | 每人成本 |
|---|---|
| 0 – 60,000 人 | $5 / 人 |
| > 60,000 人 | $25 / 人 |

- 成本為**累進階梯制**：前 60,000 人每人 $5，超過的每人 $25
- Scenario PDF 的範例不同 (0-500 免費, 501-60K $2, >60K $12)，僅供參考

---

## 遊戲架構：兩個關卡

### Level 1：預測捐款金額

- **目標變數**：`AmtThisYear`（今年捐款金額）
- **訓練資料**：`FUNDRAISE_R1_TRAIN`（100K 筆，無 Contact 變數）
- **評分資料**：`FUNDRAISE_R1_SCOREDATA`（同樣 100K 筆，時間位移：ThisYear → LastYear）
- **做法**：
  1. 在 SAS Model Studio 建立預測模型（目標 = AmtThisYear）
  2. 對 ScoreData 評分，得到每人的預測捐款金額
  3. 篩選預測金額 > 聯繫成本的 ID
  4. 匯出 CSV（僅含 ID 欄位），上傳至排行榜

### Level 2：Uplift 建模（進階）

Level 2 引入 `Contact` 變數（0/1），模擬「有聯繫 vs 無聯繫」的因果效應。

#### Level 2a — 分類模型：預測捐款機率

- **目標變數**：`GaveThisYear`（是否捐款，二元分類）
- **訓練資料**：`FUNDRAISE_R2_TRAIN`（含 Contact 欄位）
- **評分資料**：
  - `FUNDRAISE_R2_CONTACT_SCOREDATA`（Contact=1 情境）
  - `FUNDRAISE_R2_NOCONTACT_SCOREDATA`（Contact=0 情境）
- 分別對兩份 ScoreData 評分，得到：
  - P(Gave | Contact=1)
  - P(Gave | Contact=0)

#### Level 2b — 回歸模型：預測捐款金額（僅捐款者）

- **目標變數**：`AmtThisYear`（僅篩選 GaveThisYear=1 的資料訓練）
- 同樣對兩份 ScoreData 評分，得到：
  - E(Amount | Gave, Contact=1)
  - E(Amount | Gave, Contact=0)

#### Level 2c — Uplift 計算

$$EV_{contact} = P(Gave|Contact) \times E(Amt|Gave, Contact)$$
$$EV_{no\_contact} = P(Gave|NoContact) \times E(Amt|Gave, NoContact)$$
$$Uplift = EV_{contact} - EV_{no\_contact}$$

- **Uplift > 聯繫成本** → 納入聯繫名單（Persuadables）
- **Uplift ≈ 0 但 EV 高** → 不聯繫也會捐的鐵粉（Sure Things），省下成本
- **Uplift < 0** → 聯繫後反而捐更少的「沉睡犬」（Sleeping Dogs），絕對不聯繫
- **EV 都很低** → 無感者（Lost Causes），不聯繫

---

## 四象限捐款者分類

| | 不聯繫會捐 | 不聯繫不會捐 |
|---|---|---|
| **聯繫後會捐（更多）** | Sure Things（省成本） | **Persuadables（目標）** |
| **聯繫後不捐（或更少）** | **Sleeping Dogs（避開）** | Lost Causes（忽略） |

---

## 資料集總覽

共 5 份 CSV，都是同一批 100,000 人（ID 2000001–2099999）。

| 資料集 | 筆數 | 特有欄位 | 用途 |
|---|---|---|---|
| `R1_TRAIN` | 100,000 | `GaveThisYear`, `AmtThisYear` | Level 1 訓練：有答案，用來建模 |
| `R1_SCOREDATA` | 100,000 | （無 Target） | Level 1 評分：預測「下一年」，產出 ID list |
| `R2_TRAIN` | 100,000 | `Contact`, `GaveThisYear`, `AmtThisYear` | Level 2 訓練：多了 Contact 變數 |
| `R2_CONTACT_SCOREDATA` | 100,000 | `Contact=1`（無 Target） | Level 2 評分：模擬「有聯繫」情境 |
| `R2_NOCONTACT_SCOREDATA` | 100,000 | `Contact=0`（無 Target） | Level 2 評分：模擬「沒聯繫」情境 |

### 關鍵差異
- **R1 vs R2**：R2 多了 `Contact` 欄位（0/1 表示是否被聯繫），R1 沒有
- **TRAIN vs SCOREDATA**：TRAIN 有答案（`GaveThisYear`, `AmtThisYear`），SCOREDATA 沒有。SCOREDATA 是同一批人但時間往後推一年（ThisYear 變成 LastYear）
- **R2 的兩份 SCOREDATA**：同樣的人，`Contact` 分別固定為 1 和 0，讓你分別預測兩種情境下的 EV，相減得到 Uplift

### 使用時機
| 你要做什麼 | 用哪份資料 |
|---|---|
| Level 1 建模（預測金額） | `R1_TRAIN` 訓練 → `R1_SCOREDATA` 評分 |
| Level 2a 建模（預測捐款機率） | `R2_TRAIN` 訓練 → `R2_CONTACT_SCOREDATA` + `R2_NOCONTACT_SCOREDATA` 分別評分 |
| Level 2b 建模（預測捐款金額，僅捐款者） | `R2_TRAIN`（篩選 GaveThisYear=1）訓練 → 同上兩份分別評分 |
| 計算 Uplift | 合併 Level 2a + 2b 的四份評分結果 |

---

## 關鍵變數 (Data Dictionary)

| 變數名 | 說明 | 類型 |
|---|---|---|
| ID | 會員編號 | ID |
| AmtPrevYear | 前年捐款金額 | Input |
| AmtThisYear | 今年捐款金額 | **Target (Level 1)** |
| GaveThisYear | 今年是否捐款 | **Target (Level 2a)** |
| GavePrevYear | 前年是否捐款 | Input |
| Contact | 是否被聯繫 (0/1) | Input (僅 Level 2) |
| Frequency | 過去捐款次數 | Input（注意：需改 Role 為 Input） |
| Recency | 距上次捐款月數 | Input |
| Salary | 年薪 | Input |
| YearsServed | 服務年資 | Input |

> ⚠️ **重要 Metadata 設定**：
> - `Frequency` 預設 Role 可能為 "Frequency"，必須手動改為 "Input"
> - `AmtThisYear` / `GaveThisYear` 設為 Target
> - 排除與 Target 高度相關的變數（如預測 Amount 時排除 GaveThisYear）

---

## 工作流程

1. **EDA**：在 SAS Visual Analytics 探索資料分佈，確認零值膨脹程度
2. **建立 Pipeline**：
   - Level 1：一個回歸 Pipeline（Target = AmtThisYear）
   - Level 2a：一個分類 Pipeline（Target = GaveThisYear）
   - Level 2b：一個回歸 Pipeline（僅捐款者，Target = AmtThisYear）
3. **訓練與比較模型**：嘗試 Gradient Boosting、Random Forest、Neural Network 等
4. **評分 (Score)**：用 Champion Model 對 ScoreData 打分
5. **在 Dashboard 中合併**：計算 Uplift，篩選 Uplift > Cost 的 ID
6. **匯出 CSV**：僅包含 `ID` 欄位，上傳至 Cortex 排行榜
7. **迭代**：根據排行榜回饋調整模型與篩選門檻

---

## 時程

| 日期 | 事項 |
|---|---|
| 2026/3/6 (五) | Kick-off + 訓練工作坊（實體，NYU SPS） |
| 2026/3/9 週 | 比賽期間，可預約 SAS Office Hours (3/10, 3/12) |
| 2026/3/13 (五) | 最終排行榜公布 + 前十名簡報 |

---

## 排行榜與決勝負機制

### 上傳方式
- **Level 1 和 Level 2 都上傳到同一個排行榜**
- 上傳的檔案格式完全相同：**只有 `ID` 欄位的 CSV**（你決定要聯繫的人）
- 可以反覆上傳、反覆改進，每次上傳都會更新分數
- Level 1 / Level 2 只是兩種不同的建模策略，不是比賽的強制階段

### 評分方式
排行榜後台持有所有會員「實際的下一年捐款金額」（時間機器），根據你提交的 ID list 計算：

$$Operating\ Surplus = \sum(\text{你聯繫的人的實際捐款}) - \sum(\text{聯繫成本})$$

### 獎項（兩個）
| 獎項 | 評選方式 |
|---|---|
| **Best Overall Award** | 排行榜 Operating Surplus 最高的隊伍 |
| **Best Presentation** | 前十名進入簡報，由 SAS + HEC Montréal + NYU SPS 評審評選 |

### 策略自由度
- 最終產出永遠是一份 ID list CSV，排行榜只看結果，不管你用什麼方法
- 可以完全不按 Level 1 / Level 2 的教學走，自己設計策略
- 可以用 Python/R 建模，只要最後產出正確格式的 CSV 即可
- 資料來源必須用 SAS Viya 平台上提供的資料集

---

## 策略要點

1. **Level 2 是拿高分的關鍵**：Level 1 只是暖身，Level 2 的 Uplift 模型才能精準區分四種人
2. **注意成本跳漲點**：在階梯成本下，聯繫人數超過門檻後邊際成本暴增，必須在門檻前停止
3. **Sleeping Dogs 很重要**：有些人被聯繫後反而捐更少，識別並排除他們可以提升盈餘
4. **模型校準 > 模型精度**：EV 計算依賴準確的機率估計，過度擬合的模型可能產生偏差
5. **確認實際成本表**：遊戲開始後第一件事就是確認 Cost Schedule
