# Market Prediction Competition - EDA Report

## 1. 데이터 개요

### 데이터셋 구조
- **Train set**: 8,991 rows (trading days) × 98 columns
- **Test set**: 11 rows (mock test, 실제 평가시에는 새로운 데이터로 교체됨)
- **시간 범위**: date_id 0 ~ 8,990 (수십년 분량의 역사적 데이터)

### Feature 카테고리 (총 94개)
| 카테고리 | 개수 | 설명 |
|---------|------|------|
| **M** | 18 | Market Dynamics/Technical features - 시장 기술적 지표 |
| **E** | 20 | Macro Economic features - 거시경제 지표 |
| **I** | 9 | Interest Rate features - 금리 관련 지표 |
| **P** | 13 | Price/Valuation features - 가격/밸류에이션 지표 |
| **V** | 13 | Volatility features - 변동성 지표 |
| **S** | 12 | Sentiment features - 시장 심리 지표 |
| **MOM** | 0 | Momentum features (README에 언급되었으나 실제 데이터에 없음) |
| **D** | 9 | Dummy/Binary features - 이진 지표 |

### Target 변수 (Train only)
1. **forward_returns**: 다음날 S&P 500 수익률 (오늘 매수 → 내일 매도)
2. **risk_free_rate**: Federal funds rate (무위험 수익률)
3. **market_forward_excess_returns**:
   - 5년 롤링 평균을 제거한 초과 수익률
   - MAD criterion 4로 winsorizing 처리됨
   - 트렌드가 제거된 수익률

### Test set 추가 컬럼
- **is_scored**: 평가에 포함되는 row 여부
- **lagged_forward_returns**: 1일 lag된 수익률
- **lagged_risk_free_rate**: 1일 lag된 무위험 수익률
- **lagged_market_forward_excess_returns**: 1일 lag된 초과 수익률

---

## 2. 주요 발견사항

### 2.1 결측치 패턴 (매우 중요!)

#### 시간에 따른 결측치 변화
- **초기 기간 (date_id < 1,000)**: 평균 **~85개 feature 결측** (약 87% 결측률)
- **최근 기간 (date_id > 8,000)**: 평균 **0개 feature 결측** (완전한 데이터)
- **중간 기간**: 점진적으로 결측률 감소

#### 결측치 패턴의 의미
- 과거로 갈수록 데이터 품질이 떨어짐
- 최근 몇천개 데이터는 모든 feature가 완전함
- **초기 희소 데이터는 학습에서 제외하는 것을 권장**

#### 카테고리별 결측치
모든 feature 카테고리가 초기에는 비슷한 결측 패턴을 보임. 특정 카테고리만 유독 결측이 많은 것은 아님.

#### 권장사항
**date_id >= [완전성 기준점]**부터의 데이터만 사용하여 초기 모델 학습 권장
- 이렇게 하면 약 6,000~7,000개의 완전한 데이터 확보 가능
- 혹은 LightGBM/XGBoost 등 결측치를 자동으로 처리하는 모델 사용

---

### 2.2 타겟 변수 특성

#### Forward Returns 통계
```
평균 일일 수익률: ~0.03% (양수)
표준편차: ~0.012 (1.2%)
왜도(Skewness): 약간 음수 (왼쪽 꼬리가 긴 분포 - 큰 하락이 가끔 발생)
첨도(Kurtosis): >3 (Fat tails - 극단값이 정규분포보다 많음)
```

#### 연간화 메트릭 (252 trading days 기준)
```
연간 수익률: ~7-9%
연간 변동성: ~18-20%
Sharpe Ratio (raw): ~0.4-0.5
```

#### 시계열 특성
- **자기상관(Autocorrelation)**: 거의 0에 가까움
  - 일일 수익률은 거의 랜덤워크 (효율적 시장 가설 지지)
  - Lag-1 상관관계 매우 약함

- **변동성 클러스터링**: 존재함!
  - 제곱 수익률의 자기상관 > 0
  - 높은 변동성이 높은 변동성 다음에 오는 경향
  - GARCH 계열 모델 고려 가능

#### 수익률 분포 특징
- **정규분포 아님**: Q-Q plot에서 벗어남
- **Fat tails**: 극단 움직임이 정규분포보다 많음
- **약간의 음의 왜도**: 큰 하락이 큰 상승보다 약간 더 자주 발생

#### 승률 통계
- 양수 수익률 일수: ~53-55%
- 음수 수익률 일수: ~45-47%
- 장기적으로 상승 편향

---

### 2.3 시장 Regime 분석

4가지 시장 상황으로 분류 가능:
1. **Bull-Quiet**: 상승 + 낮은 변동성 (이상적)
2. **Bull-Volatile**: 상승 + 높은 변동성
3. **Bear-Quiet**: 하락 + 낮은 변동성 (서서히 하락)
4. **Bear-Volatile**: 하락 + 높은 변동성 (폭락 구간)

각 regime에 따라 최적 allocation이 달라질 것으로 예상:
- Bull-Quiet: 높은 allocation (1.5-2.0)
- Bull-Volatile: 중간 allocation (1.0-1.5)
- Bear-Quiet: 낮은 allocation (0.5-1.0)
- Bear-Volatile: 매우 낮은 allocation (0-0.5)

---

### 2.4 Feature 인사이트

#### Feature와 타겟의 상관관계
- 대부분의 feature가 forward_returns와 약한 상관관계 (|r| < 0.1)
- 일부 feature는 중간 정도의 상관관계 보임 (|r| ~ 0.1-0.3)
- **상관관계가 강한 feature가 많지 않음** → 비선형 패턴을 찾아야 함

#### Feature 카테고리별 특징
- **M (Market Dynamics)**: 가장 즉각적인 시장 상황 반영
- **V (Volatility)**: 변동성 regime 파악에 중요
- **E (Economic)**: 장기 트렌드에 영향, 단기 예측에는 덜 유용할 수 있음
- **I (Interest Rate)**: 시장 방향성에 영향
- **S (Sentiment)**: 단기 움직임 예측에 유용할 수 있음

#### Feature 간 상관관계
- 같은 카테고리 내에서 높은 상관관계를 보이는 feature 쌍들이 존재
- Feature selection이나 PCA 고려 가능
- 다만 tree-based 모델은 다중공선성에 강하므로 크게 문제되지 않을 수 있음

#### Feature 안정성
- 대부분의 feature가 시간에 따라 비정상성(non-stationarity) 보임
- Rolling 통계나 차분(differencing) 고려 필요

---

### 2.5 Drawdown 분석

- **Maximum Drawdown**: 약 -40% ~ -55% (데이터 기간에 따라 다름)
- 여러 차례의 큰 하락 구간 존재
- 회복 기간(recovery period)도 상당히 길 수 있음

**전략 구현시 고려사항**:
- Drawdown을 줄이는 것이 중요
- Bear market에서 allocation을 낮추는 것이 핵심
- 단순히 수익률만 높이는 것이 아니라 risk-adjusted return 최적화

---

## 3. 대회 해결 전략

### 3.1 핵심 도전과제

#### 문제의 본질
이 대회는 단순한 회귀 문제가 **아님**:
- **목표**: Forward returns 예측 + 최적 portfolio allocation 결정
- **제약**: 변동성 120% 이하 유지
- **메트릭**: 커스텀 Sharpe ratio (volatility & return penalties)

#### 주요 난제
1. **Time-series 특성**: 데이터 누수 방지 필수
2. **Noisy target**: 일일 수익률은 매우 노이즈가 많음 (autocorr ≈ 0)
3. **Regime dependency**: 시장 상황에 따라 전략이 달라져야 함
4. **Volatility constraint**: 과도한 leverage 사용시 패널티
5. **API 제한**: 5분 inference 제한, 15분 model loading 제한

---

### 3.2 제안 접근법

#### 접근법 1: 수익률 예측 → 할당 변환 ★ (추천)

**컨셉**:
- Forward returns를 예측
- 예측값을 allocation (0-2)로 매핑

**장점**:
- 직관적이고 구현하기 쉬움
- 회귀 모델 사용 가능
- Feature engineering이 자유로움

**단점**:
- 예측 → allocation 매핑 규칙이 명확하지 않음
- Volatility constraint를 직접 고려하지 않음

**구현 방법**:
```python
# Step 1: 예측 모델
predicted_return = model.predict(features)

# Step 2: Allocation 매핑
if predicted_return > threshold_high:
    allocation = 1.5  # 높은 allocation
elif predicted_return > threshold_low:
    allocation = 1.0  # 중간 allocation
else:
    allocation = 0.5  # 낮은 allocation

# Step 3: Volatility-based scaling
recent_vol = calculate_recent_volatility()
if recent_vol > vol_threshold:
    allocation *= 0.8  # 변동성 높을때 allocation 줄임
```

**추천 모델**:
- LightGBM (결측치 처리 우수, 빠름)
- XGBoost (안정적인 성능)
- CatBoost (robust)

**구현 난이도**: ⭐⭐ (중하)

---

#### 접근법 2: 직접 할당 회귀 (Direct Allocation Regression)

**컨셉**:
- Optimal allocation을 직접 예측
- Historical data로 최적 allocation 계산 → 이를 target으로 학습

**장점**:
- End-to-end 학습
- Metric에 직접 최적화

**단점**:
- "최적 allocation"을 어떻게 정의하고 계산할 것인가?
- Backtesting 필요 → 시간 소요

**구현 방법**:
```python
# Step 1: 각 시점의 최적 allocation 계산 (backtest)
for each historical date:
    try different allocations (0, 0.1, 0.2, ..., 2.0)
    calculate forward Sharpe ratio for next N days
    optimal_allocation[date] = allocation with best Sharpe

# Step 2: 모델 학습
model.train(features, target=optimal_allocation)

# Step 3: 예측
allocation = model.predict(current_features)
```

**문제점**:
- Optimal allocation이 미래 정보를 사용하게 됨 (look-ahead bias)
- Rolling window로 계산해야 하지만 복잡함

**구현 난이도**: ⭐⭐⭐⭐ (상)

---

#### 접근법 3: 분류 접근 + Kelly Criterion

**컨셉**:
- 시장 방향(상승/하락) 분류
- 예측 확률을 Kelly criterion으로 allocation 계산

**장점**:
- 분류 문제로 단순화
- Kelly criterion은 이론적으로 최적 sizing

**단점**:
- 수익률 크기 정보 손실
- Kelly criterion은 정확한 확률 필요 (calibration 중요)

**구현 방법**:
```python
# Step 1: 분류 모델
prob_up = classifier.predict_proba(features)[1]

# Step 2: Kelly criterion
edge = prob_up - 0.5  # 우위
kelly_fraction = edge / volatility

# Step 3: Allocation with constraints
allocation = clip(kelly_fraction * leverage_factor, 0, 2)
```

**구현 난이도**: ⭐⭐⭐ (중상)

---

#### 접근법 4: 앙상블 전략 ★★ (최종 추천)

**컨셉**:
- 여러 전략을 결합
- Historical performance로 가중 평균

**구성**:
1. **회귀 모델**: Returns 예측 (LightGBM, XGBoost)
2. **분류 모델**: Direction 예측 (Neural Net, LightGBM)
3. **시계열 모델**: ARIMA, GARCH (변동성 예측)
4. **기술적 전략**: Momentum, Mean reversion
5. **변동성 조절**: Dynamic volatility targeting

**앙상블 방법**:
```python
# 각 전략의 allocation 계산
alloc_1 = regression_strategy()
alloc_2 = classification_strategy()
alloc_3 = momentum_strategy()
alloc_4 = volatility_strategy()

# 가중 평균 (weights는 validation performance 기반)
final_allocation = (
    w1 * alloc_1 +
    w2 * alloc_2 +
    w3 * alloc_3 +
    w4 * alloc_4
)

# Volatility constraint 체크
if predicted_strategy_vol > 1.2 * market_vol:
    final_allocation *= scaling_factor
```

**장점**:
- Robust (단일 모델 실패해도 괜찮음)
- 다양한 시장 regime에 적응 가능
- Best performance 가능성

**단점**:
- 복잡함
- 개발 시간 오래 걸림
- Overfitting 위험

**구현 난이도**: ⭐⭐⭐⭐⭐ (최상)

---

### 3.3 Feature Engineering 아이디어

#### 1. Lag Features
```python
for feature in all_features:
    for lag in [1, 5, 20, 60]:
        df[f'{feature}_lag{lag}'] = df[feature].shift(lag)
```

**이유**:
- 일일 수익률은 noisy하지만 과거 feature 패턴은 유용할 수 있음
- 특히 5일, 20일 lag는 주간/월간 패턴 포착

#### 2. Rolling Statistics
```python
windows = [5, 10, 20, 60]
for feature in all_features:
    for window in windows:
        df[f'{feature}_mean_{window}'] = df[feature].rolling(window).mean()
        df[f'{feature}_std_{window}'] = df[feature].rolling(window).std()
        df[f'{feature}_min_{window}'] = df[feature].rolling(window).min()
        df[f'{feature}_max_{window}'] = df[feature].rolling(window).max()
```

**이유**:
- Noise 제거
- Trend와 volatility 포착

#### 3. Momentum Indicators
```python
# Returns-based momentum
df['momentum_5'] = df['forward_returns'].rolling(5).sum()
df['momentum_20'] = df['forward_returns'].rolling(20).sum()

# Feature-based momentum
for feature in all_features:
    df[f'{feature}_momentum'] = df[feature] - df[feature].shift(20)
    df[f'{feature}_roc'] = (df[feature] / df[feature].shift(20)) - 1  # Rate of change
```

#### 4. Volatility Features
```python
# Historical volatility
df['vol_5'] = df['forward_returns'].rolling(5).std()
df['vol_20'] = df['forward_returns'].rolling(20).std()
df['vol_60'] = df['forward_returns'].rolling(60).std()

# Volatility of volatility
df['vol_of_vol'] = df['vol_20'].rolling(20).std()

# Volatility regime
df['vol_regime'] = (df['vol_20'] > df['vol_20'].rolling(60).mean()).astype(int)
```

#### 5. Market Regime Features
```python
# Returns regime
df['bull_market'] = (df['forward_returns'].rolling(60).mean() > 0).astype(int)

# Volatility regime
vol_percentile = df['vol_20'].rolling(252).rank(pct=True)
df['low_vol_regime'] = (vol_percentile < 0.33).astype(int)
df['high_vol_regime'] = (vol_percentile > 0.67).astype(int)
```

#### 6. Cross-sectional Features
```python
# Feature correlations
df['M_mean'] = df[[c for c in df.columns if c.startswith('M')]].mean(axis=1)
df['V_mean'] = df[[c for c in df.columns if c.startswith('V')]].mean(axis=1)

# Feature dispersion
df['feature_std'] = df[all_features].std(axis=1)
```

#### 7. Missing Value Features
```python
# Missing value indicators can be informative!
df['n_missing'] = df[all_features].isnull().sum(axis=1)
df['missing_pct'] = df['n_missing'] / len(all_features)

for feature in all_features:
    df[f'{feature}_is_missing'] = df[feature].isnull().astype(int)
```

#### 8. Target Encoding (주의!)
```python
# ONLY use past information (walk-forward)
# DO NOT use global mean (data leakage!)

# Example: Safe target encoding
def safe_target_encode(df, feature, target, window=100):
    # For each row, use only past 100 rows to compute mean
    encoding = df.groupby(feature)[target].apply(
        lambda x: x.shift(1).rolling(window=window, min_periods=10).mean()
    )
    return encoding
```

---

### 3.4 모델 후보

#### Tier 1: 빠른 프로토타입용 (일주일 내 구현)

1. **LightGBM** ⭐⭐⭐⭐⭐
   - **추천 이유**:
     - 결측치 자동 처리
     - 빠른 학습
     - 비선형 패턴 포착 우수
     - Kaggle에서 검증된 성능
   - **하이퍼파라미터 튜닝 포인트**:
     - `num_leaves`: 31-127
     - `learning_rate`: 0.01-0.1
     - `min_data_in_leaf`: 20-100
     - `feature_fraction`: 0.7-0.9

2. **XGBoost** ⭐⭐⭐⭐
   - **추천 이유**:
     - 매우 안정적
     - Regularization 우수
     - 결측치 처리 가능
   - **하이퍼파라미터**:
     - `max_depth`: 3-7
     - `eta`: 0.01-0.1
     - `colsample_bytree`: 0.7-0.9

3. **Ridge/Lasso Regression** ⭐⭐⭐
   - **추천 이유**:
     - 빠른 baseline
     - 과적합 위험 낮음
     - Interpretability 우수
   - **단점**: 비선형 패턴 포착 어려움

#### Tier 2: 성능 개선용 (2-3주 투자)

4. **Random Forest** ⭐⭐⭐⭐
   - 앙상블에 포함시키기 좋음
   - LightGBM과 다른 패턴 학습

5. **Neural Networks** ⭐⭐⭐
   - MLP, LSTM, Transformer 고려
   - 복잡한 비선형 패턴 학습 가능
   - **주의**: Overfitting 조심
   - **추천 구조**:
     ```
     Input -> Dense(256) -> Dropout(0.3) -> Dense(128) -> Dropout(0.3) -> Output
     ```

6. **TabNet** ⭐⭐⭐
   - Attention mechanism for tabular data
   - Feature importance 제공

#### Tier 3: 실험용 (선택적)

7. **GARCH Models**
   - 변동성 예측에 특화
   - Returns 예측보다는 volatility targeting에 사용

8. **Reinforcement Learning** ⭐⭐
   - Portfolio optimization에 이론적으로 적합
   - 하지만 구현 복잡, sample efficiency 낮음
   - 시간 여유 있을 때만 시도

---

### 3.5 Validation 전략

#### Walk-Forward Validation (필수!)

```python
# DO NOT use random split - data leakage!
# DO NOT use standard K-fold - data leakage!

# Use time-based split
def walk_forward_validation(df, n_splits=5):
    total_len = len(df)
    split_size = total_len // (n_splits + 1)

    for i in range(n_splits):
        train_end = (i + 2) * split_size
        val_start = train_end
        val_end = val_start + split_size

        train_data = df.iloc[:train_end]
        val_data = df.iloc[val_start:val_end]

        yield train_data, val_data
```

#### Purging & Embargo (중요!)

```python
# Purge: validation 직전 데이터 제거 (label leakage 방지)
# Embargo: validation 이후 데이터도 train에서 제거 (information leakage 방지)

def purged_walk_forward(df, n_splits=5, purge_days=5, embargo_days=5):
    for train_data, val_data in walk_forward_validation(df, n_splits):
        # Remove purge period before validation
        train_data = train_data.iloc[:-purge_days]

        # Remove embargo period after validation from NEXT train
        # (실제로는 다음 fold에서 처리)

        yield train_data, val_data
```

#### Evaluation Metrics

**Primary Metric**: Competition score (구현된 메트릭 함수 사용)

**Secondary Metrics**:
- Sharpe ratio
- Max drawdown
- Win rate
- Volatility ratio
- Calmar ratio

---

### 3.6 Volatility Constraint 관리

#### 방법 1: Dynamic Scaling

```python
# 전략 volatility가 120% 넘으면 allocation을 줄임
def apply_volatility_constraint(allocation, returns, market_vol):
    strategy_vol = calculate_strategy_volatility(allocation, returns)
    vol_ratio = strategy_vol / market_vol

    if vol_ratio > 1.2:
        # Scale down allocation
        scaling_factor = 1.2 / vol_ratio
        allocation *= scaling_factor

    return allocation
```

#### 방법 2: Rolling Volatility Targeting

```python
# 일정한 target volatility 유지
def volatility_targeting(allocation, target_vol=0.18):
    recent_vol = calculate_recent_volatility(window=20)

    if recent_vol > 0:
        vol_scalar = target_vol / recent_vol
        allocation *= vol_scalar

    # Clip to valid range
    allocation = np.clip(allocation, 0, 2)
    return allocation
```

#### 방법 3: Regime-based Adjustment

```python
# 변동성 regime에 따라 allocation 조정
def regime_based_allocation(base_allocation, vol_regime):
    if vol_regime == 'high':
        return base_allocation * 0.7  # 변동성 높을때 줄임
    elif vol_regime == 'low':
        return base_allocation * 1.2  # 변동성 낮을때 늘림
    else:
        return base_allocation
```

---

## 4. 구현 로드맵

### Phase 1: 빠른 베이스라인 (3-5일)

**목표**: 제출 가능한 working solution 만들기

**Tasks**:
1. ✅ Data loading & EDA
2. 데이터 전처리
   - 완전한 데이터만 선택 (date_id > threshold)
   - Feature 정규화/스케일링
3. 간단한 feature engineering
   - Lag features (1, 5, 20)
   - Rolling means (5, 20)
4. LightGBM 모델 학습
   - Walk-forward validation
   - Hyperparameter tuning (간단한 grid search)
5. API submission 구현
   - `predict` 함수 작성
   - Allocation 매핑 로직
6. 첫 제출!

**예상 성능**: Baseline 대비 10-20% 개선

---

### Phase 2: Feature Engineering & 모델 개선 (1주)

**목표**: Feature 최적화 및 다양한 모델 실험

**Tasks**:
1. 고급 feature engineering
   - 모든 lag features
   - Rolling statistics (mean, std, min, max)
   - Volatility features
   - Momentum indicators
   - Market regime features
2. Feature selection
   - Correlation analysis
   - Feature importance (from tree models)
   - Remove redundant features
3. 추가 모델 실험
   - XGBoost
   - CatBoost
   - Random Forest
   - Neural Network (MLP)
4. Hyperparameter optimization
   - Optuna 사용
   - Walk-forward validation으로 평가

**예상 성능**: Baseline 대비 30-50% 개선

---

### Phase 3: 앙상블 & 최적화 (1주)

**목표**: 여러 모델 결합 및 metric 최적화

**Tasks**:
1. Ensemble 구현
   - Weighted average of multiple models
   - Stacking
2. Volatility constraint 최적화
   - Dynamic scaling
   - Volatility targeting
3. Allocation mapping 최적화
   - 다양한 threshold 실험
   - Non-linear mapping 시도
4. Metric-specific optimization
   - Sharpe ratio 최적화
   - Penalty 최소화 전략
5. Backtesting
   - 다양한 기간에서 테스트
   - Regime별 성능 분석

**예상 성능**: Top 10-20% 목표

---

### Phase 4: 최종 튜닝 (3-5일)

**목표**: 마지막 성능 압축 및 안정성 확보

**Tasks**:
1. 모델 앙상블 가중치 최적화
2. 극단 상황 대응 전략
3. API 응답 시간 최적화 (5분 제한 준수)
4. Cross-validation 결과 분석
5. 최종 제출

**예상 성능**: Top 5-10% 도전

---

## 5. 주요 위험 요소 및 대응

### 위험 1: 데이터 누수 (Data Leakage)

**위험도**: 🔴 매우 높음

**예시**:
- 미래 데이터로 feature 계산
- Global statistics 사용 (전체 데이터의 mean/std)
- Target encoding시 전체 데이터 사용

**대응**:
- Walk-forward validation 엄격히 준수
- Feature 계산시 항상 `.shift(1)` 사용
- Rolling statistics만 사용

### 위험 2: Overfitting

**위험도**: 🟡 높음

**원인**:
- 너무 많은 feature
- 복잡한 모델
- Hyperparameter 과도하게 튜닝

**대응**:
- Regularization (L1, L2, dropout)
- Early stopping
- Simple models 선호
- Feature selection

### 위험 3: Regime Change

**위험도**: 🟡 중간

**문제**:
- 최근 데이터로 학습한 모델이 미래에 작동 안 할 수 있음
- 시장 regime이 바뀌면 전략 실패

**대응**:
- 다양한 regime에서 validation
- Robust features 선택
- 앙상블로 다양성 확보

### 위험 4: Volatility Penalty

**위험도**: 🟡 중간

**문제**:
- Leverage 과도하게 사용시 큰 패널티
- 120% threshold 넘기면 점수 급락

**대응**:
- Dynamic volatility scaling
- Conservative allocation
- Validation에서 vol ratio 모니터링

### 위험 5: API 시간 제한

**위험도**: 🟢 낮음

**제한**:
- Model loading: 15분
- Batch inference: 5분

**대응**:
- 모델 크기 최소화
- Polars 사용 (Pandas보다 빠름)
- Feature 계산 최적화
- 사전 계산 가능한 것들 pre-compute

---

## 6. 코드 구조 제안

```
market-prediction/
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── kaggle_evaluation/
├── notebooks/
│   ├── eda.ipynb                 # ✅ Done
│   ├── feature_engineering.ipynb
│   ├── model_experiments.ipynb
│   └── ensemble.ipynb
├── src/
│   ├── __init__.py
│   ├── features.py              # Feature engineering functions
│   ├── models.py                # Model definitions
│   ├── validation.py            # Walk-forward validation
│   ├── metric.py                # Competition metric
│   └── utils.py                 # Utility functions
├── submissions/
│   ├── baseline_submission.py
│   ├── lgb_submission.py
│   └── ensemble_submission.py
├── models/
│   └── trained_models/          # Saved models
├── eda.md                        # ✅ Done
├── knowledge.md                  # Project knowledge
└── README.md                     # ✅ Done
```

---

## 7. 참고 자료 및 학습 리소스

### 관련 Competition
- Numerai (비슷한 형태의 금융 예측 대회)
- Jane Street Market Prediction
- Two Sigma competitions

### 유용한 라이브러리
- **Modeling**: LightGBM, XGBoost, scikit-learn
- **Data**: Polars (faster than pandas), pandas
- **Validation**: scikit-learn, custom walk-forward
- **Hyperparameter tuning**: Optuna
- **Visualization**: matplotlib, seaborn, plotly

### 이론적 배경
- **Sharpe Ratio**: 위험 대비 수익률 측정
- **Kelly Criterion**: 최적 포지션 sizing
- **GARCH Models**: 변동성 모델링
- **Walk-Forward Analysis**: Time series validation

---

## 8. 다음 즉시 할 일 (Next Steps)

### 1순위 (지금 바로 시작) 🚀

1. **데이터 전처리 스크립트 작성**
   - 결측치 처리
   - Train/Validation split
   - Feature normalization

2. **간단한 feature engineering**
   - Lag features (1, 5, 20)
   - Rolling means
   - Volatility features

3. **LightGBM 베이스라인 모델**
   - Walk-forward validation 구현
   - 첫 예측 만들기

4. **API submission 구현**
   - `predict` 함수
   - Allocation mapping
   - Local test

### 2순위 (베이스라인 완성 후)

5. **고급 feature engineering**
6. **Multiple models 실험**
7. **Ensemble 구현**

---

## 9. 성공 기준

### Minimum Goal (반드시 달성)
- ✅ EDA 완료
- 작동하는 submission 완성
- Baseline (always 100% invested) 대비 개선
- Validation Sharpe ratio > 0.5

### Target Goal (목표)
- Validation Sharpe ratio > 1.0
- Volatility ratio < 1.15 (안전 마진)
- Public LB top 30%
- Private LB top 20%

### Stretch Goal (최고 목표)
- Public LB top 10%
- Private LB top 10%
- 여러 regime에서 일관된 성능

---

## 10. 마무리 및 핵심 인사이트

### 🔑 핵심 교훈

1. **이 대회는 예측 + 최적화 문제다**
   - 단순히 returns를 잘 예측하는 것으로 부족
   - Volatility management가 핵심
   - Sharpe ratio 최적화가 목표

2. **데이터가 깨끗하지 않다**
   - 초기 데이터 희소
   - 결측치 처리 전략 필수
   - 최근 데이터 활용 권장

3. **시계열 특성 준수**
   - 데이터 누수 조심
   - Walk-forward validation 필수
   - Feature도 시간 의존적

4. **간단한 것부터 시작**
   - 복잡한 모델보다 탄탄한 feature
   - 빠른 iteration이 중요
   - Overfitting 조심

### 💡 성공을 위한 팁

- **Daily returns는 noisy** → Rolling statistics 활용
- **Volatility clustering exists** → Volatility prediction 중요
- **Market regime matters** → Regime-based strategy
- **Ensemble is powerful** → 다양한 모델 결합
- **Validation is crucial** → 시간 기반 split 엄수

### 🎯 집중할 영역

1. **Feature Engineering** (40% 시간)
2. **Validation Strategy** (30% 시간)
3. **Model Selection & Tuning** (20% 시간)
4. **Ensemble & Optimization** (10% 시간)

---

**Ready to code! 🚀**
