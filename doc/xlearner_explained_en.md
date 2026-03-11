# X-Learner: Complete Explanation

## 1. Why Do We Need X-Learner?

### Background

Our goal is simple: **decide who to contact in order to maximize the net profit of the fundraising campaign.**

Intuitively, the best strategy is to "only contact people who donate more because they were contacted." But there is a fundamental problem:

> For each person, we can only observe either the outcome *with* contact or the outcome *without* contact — **never both at the same time**.

This is known as the **Counterfactual Problem** in causal inference.

---

### The Reality of Training Data

In R2_TRAIN, 100K people are split into two groups:

| Group | Size | Description |
|-------|------|-------------|
| Contact=0 (not contacted) | 89,886 | Large group, but never contacted |
| Contact=1 (contacted) | 10,114 | Small group, but we have post-contact outcomes |

The problem: **Contact=1 has only 10K records. Training a "post-contact behavior model" on 10K samples is prone to severe overfitting.**

We tested T-Learner (which trains separate models for each group), and M1's Holdout AUC was only 0.59 — barely better than random guessing.

---

## 2. The Core Idea of X-Learner

**Borrow strength from the large group (89K) to better understand the small group (10K).**

For each contacted person, we ask:

> "What would this person have donated *without* being contacted?"

We don't have this answer directly, but we can estimate it using the model trained on the 89K uncontacted people. This estimate is far more stable than anything learned from 10K samples alone.

---

## 3. Four Steps in Detail

### Step 1: Train Two Base Models

Split the data by Contact and train one model per group:

**M0 (Control Model)**
- Training data: 71,872 people with Contact=0 (80% train split)
- Features: demographics + donation history (19 features, **Contact excluded**)
- Target: `GaveThisYear` (0 or 1)
- Meaning: "Given no contact, how likely is this person to donate?"
- Result: Holdout AUC = 0.6615 (stable due to large sample size)

**M1 (Treatment Model)**
- Training data: 8,128 people with Contact=1 (80% train split)
- Features: same 19 features (**Contact excluded**)
- Target: `GaveThisYear` (0 or 1)
- Meaning: "After being contacted, how likely is this person to donate?"
- Result: Holdout AUC = 0.5901 (unstable due to small sample — this is exactly what X-Learner addresses)

---

### Step 2: Compute Pseudo Outcomes

This is the most important and innovative step.

**For contacted people (Contact=1): compute D1**

```
D1 = Actual outcome (Y1) − M0's predicted probability of donating without contact
```

Plain English:
- M0 says: "If this person hadn't been contacted, there's a 20% chance they'd donate"
- The person was actually contacted and donated (Y1 = 1)
- D1 = 1 − 0.20 = **+0.80** (being contacted caused them to donate 80% more than expected → strong Persuadable signal)

| D1 value | Meaning |
|----------|---------|
| D1 > 0 | Contact caused this person to donate more than expected → Persuadable signal |
| D1 ≈ 0 | Contact had no effect → Lost Cause |
| D1 < 0 | Contact caused this person to donate *less* than expected → Sleeping Dog signal |

**Why is this meaningful?**
M0 was trained on 89K people and provides a reliable counterfactual baseline. D1 is not computed from the 10K alone — it leverages M0's stable estimate, bypassing the small-sample problem.

---

**For uncontacted people (Contact=0): compute D0**

```
D0 = M1's predicted probability if contacted − Actual outcome (Y0)
```

Plain English:
- M1 says: "If we contacted this person, there's a 40% chance they'd donate"
- The person was not contacted and did not donate (Y0 = 0)
- D0 = 0.40 − 0 = **+0.40** (contacting them would increase donation probability by 40%)

---

### Step 3: Train CATE Estimators

After Step 2, we have:
- D1 values for each of the 10K contacted people
- D0 values for each of the 90K uncontacted people

But at scoring time, we need to estimate Uplift for **100K completely new people**. So we train two "Uplift predictors":

**τ1 (Uplift predictor from the treated group's perspective)**
- Training data: 8,128 Contact=1 people
- Features: same 19 features
- Target: **D1** (a continuous Uplift value, not a binary label)
- Meaning: for a new person, estimate their individual treatment effect from the treatment group's perspective
- Result: Train R² = 0.3139 (much better than the ~0 R² from the regression model, because D1 incorporates M0's counterfactual information)

**τ0 (Uplift predictor from the control group's perspective)**
- Training data: 71,872 Contact=0 people
- Features: same 19 features
- Target: **D0**
- Meaning: for a new person, estimate their individual treatment effect from the control group's perspective
- Result: Train R² = 0.2450

---

### Step 4: Propensity Score Weighted Combination

We now have two Uplift estimates. Which one should we trust more?

The issue: Contact=1 people were **not randomly selected**. People with prior donation history or higher engagement were more likely to be contacted in the first place. This creates selection bias between the two groups.

Solution: train a **Propensity Score model** g(x):

**Propensity Score Model**
- Training data: all 80,000 people (train split)
- Features: same 19 features
- Target: `Contact` (0 or 1)
- Meaning: "Given this person's features, how likely were they to be contacted?"
- Result: Train AUC = 0.7210 (confirms Contact is not randomly assigned, so PS weighting is meaningful)

Use g(x) to determine the weighting:

```
Final CATE = g(x) × τ0(x) + (1 − g(x)) × τ1(x)
```

| g(x) | Meaning | Weighting |
|------|---------|-----------|
| High (e.g., 0.8) | This person closely resembles those who tend to get contacted | Trust τ0 more (control group perspective is more neutral) |
| Low (e.g., 0.1) | This person rarely gets contacted | Trust τ1 more (treatment group perspective is more relevant) |

---

## 4. All Models at a Glance

| Model | Training Data | # Features | Target | Algorithm |
|-------|--------------|------------|--------|-----------|
| M0 | Contact=0, 71,872 people | 19 | GaveThisYear (0/1) | GBM (Optuna best params) |
| M1 | Contact=1, 8,128 people | 19 | GaveThisYear (0/1) | GBM |
| τ0 | Contact=0, 71,872 people | 19 | D0 (continuous) | GBM |
| τ1 | Contact=1, 8,128 people | 19 | D1 (continuous) | GBM |
| Propensity | Full sample, 80,000 people | 19 | Contact (0/1) | GBM |

Hyperparameters (same for all models, from Optuna search):
- n_estimators=375, max_depth=6, learning_rate=0.0129
- subsample=0.604, min_samples_leaf=45

---

## 5. The 19 Features

| Feature | Description |
|---------|-------------|
| Woman | Gender (0/1) |
| Age | Age |
| Salary | Annual salary |
| SeniorList | Senior list membership |
| NbActivities | Number of activities attended |
| Referrals | Number of referrals made |
| Recency | Months since last donation |
| Frequency | Number of past donations |
| Seniority | Years of membership |
| TotalGift | Total cumulative donation amount |
| MinGift | Smallest single donation |
| MaxGift | Largest single donation |
| GaveLastYear | Donated last year (0/1) |
| AmtLastYear | Donation amount last year |
| Education_High School | High school education (one-hot) |
| Education_University / College | University/college education (one-hot) |
| City_Downtown | Lives downtown (one-hot) |
| City_Rural | Lives in rural area (one-hot) |
| City_Suburban | Lives in suburbs (one-hot) |

> ⚠️ **Contact is NOT a feature**: X-Learner handles Contact through group splitting and counterfactual estimation, not as an input feature.

---

## 6. Scoring Pipeline

After training all 5 models, we score 100K new people in the SCOREDATA:

```
1. X = preprocess(SCOREDATA)          ← compute 19 features
2. g  = Propensity.predict(X)          ← each person's propensity to be contacted
3. t1 = τ1.predict(X)                  ← Uplift from treatment group perspective
4. t0 = τ0.predict(X)                  ← Uplift from control group perspective
5. CATE = g × t0 + (1 − g) × t1       ← weighted combination
6. uplift_dollar = CATE × $67.25       ← multiply by donor mean to convert to dollar terms
7. if uplift_dollar > contact_cost → add to contact list
```

---

## 7. Results

| Metric | Value |
|--------|-------|
| Validation strategy efficiency | **84.9%** (vs 70.3% for GBM Single Model) |
| SCOREDATA Persuadables | **60,000 people** |
| Estimated cost | $300,000 (60K × $5, right at the cost tier boundary) |
| Leaderboard Surplus | **$8,718,670** (best overall) |

---

## 8. X-Learner vs Level 2 Single Model

| Dimension | Level 2 Single Model | X-Learner |
|-----------|---------------------|-----------|
| **How Contact is handled** | Treated as a regular feature; model learns a global Contact effect | Group splitting + counterfactual estimation; individual-level Uplift |
| **Question asked** | "What is this person's donation rate *when* contacted?" | "**How much more** does this person donate *because of* contact?" |
| **Class imbalance** | Contact=1 is only 10%, signal diluted by 90% | Large group (90K) knowledge borrowed to support small group (10K) |
| **Number of models** | 2 (classifier + regressor) | 5 (M0+M1+τ0+τ1+Propensity) |
| **Individualization** | Low (all people share the same Contact effect estimate) | High (each person has their own D1/D0 → τ estimate) |
| **Validation efficiency** | 70.3% | **84.9%** |
| **Leaderboard Surplus** | $8,688,900 | **$8,718,670** |
| **Difference** | — | **+$29,770 (+0.34%)** |

---

### The Fundamental Difference in How Uplift is Computed

**Level 2 Single Model:**
```
One model, predict twice, take the difference

Model input [x, Contact=1] → P1
Model input [x, Contact=0] → P0
Uplift = P1 − P0
```
Contact is just an ordinary feature. The model learns a **global rule**: "people with Contact=1 donate at a rate 22% higher than those with Contact=0" — applied roughly the same way to everyone.

**X-Learner:**
```
No Contact feature. Instead, use counterfactual differences to directly
estimate each person's individual treatment effect.

D1 = actual outcome (with contact) − estimated outcome (without contact, via M0)
D0 = estimated outcome (with contact, via M1) − actual outcome (without contact)
CATE = Uplift estimator trained on D1/D0
```
Each person's D1/D0 is computed based on their own situation, not a global rule.

---

### Side-by-Side Example: Same Person, Two Approaches

Assume: "55-year-old female, AmtLastYear=$0, NbActivities=2"

**Level 2 Single Model:**
```
1. Feed into model with Contact=1 → P1 = 0.35 (35% donation rate)
2. Feed into model with Contact=0 → P0 = 0.13 (13% donation rate)
3. Uplift_prob = 0.35 − 0.13 = 0.22
4. Uplift_dollar = 0.22 × E_amount_model  (model also predicts donation amount)
```

**X-Learner:**
```
1. M0 (trained on 89K control group) predicts her donation rate without contact = 0.15
2. She was actually contacted and donated (Y1 = 1)
   → D1 = 1 − 0.15 = +0.85 (contact caused 85% more than expected)
3. τ1 learns: for people with these features, CATE ≈ 0.18 after contact
4. Final CATE = 0.18 (after Propensity Score weighting)
```

**Converting to dollar terms and making the contact decision:**

X-Learner multiplies CATE by the **mean donation amount across all donors** ($67.25):
```
uplift_dollar = CATE × $67.25
             = 0.18 × $67.25
             = $12.11  >  contact cost $5 → Add to contact list (Persuadable)
```

| Step | Level 2 | X-Learner |
|------|---------|-----------|
| Number of classifiers | 1 (Contact is a feature) | 2 (M0 + M1, separately trained) |
| Uplift estimation | Same model predicts twice, take diff | D1/D0 counterfactuals → τ estimators |
| Amount estimation | Regression model predicts E(amt) | Fixed donor mean $67.25 |
| Final Uplift | P_diff × E_amt | CATE × $67.25 |

---

### Why Use the Full Mean $67.25 Instead of a Trimmed Mean?

Donor amounts are heavily right-skewed (median $25, mean $67, max $10,000). We ran experiments with trimmed means:

| Cutoff | Mean | People Selected | Validation Surplus | Efficiency |
|--------|------|----------------|-------------------|------------|
| **No trimming (adopted)** | **$67.25** | **14,878** | **$147,240** | **84.9%** |
| Remove top 1% (>$750) | $49.25 | 13,910 | $139,040 | 80.2% |
| Remove top 5% (>$200) | $36.46 | 12,537 | $126,520 | 73.0% |
| Remove top 10% (>$100) | $31.03 | 11,640 | $122,355 | 70.6% |

**Conclusion: more trimming → lower Surplus.** Although large donors ($750+) represent only 1% of cases, their contributions are real and should not be excluded. The arithmetic mean is the correct statistic for expected value calculations.

---

### In One Sentence

> Level 2 asks: "How likely is this person to donate when contacted?"
>
> X-Learner asks: "**How much more does this person donate because they were contacted? Is that increment worth the $5 contact cost?**"
>
> The latter question directly targets the business objective.
