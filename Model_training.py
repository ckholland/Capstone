#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Load in the data
import pandas as pd

file = "pendulum_data.csv"

df = pd.read_csv(file)
print("Loaded:", file)
print("Shape:", df.shape)
df.head()


# In[5]:


# Keep only the necessary columns
desired_cols = ['company', 'upload_date', 'impression_count_comb', 'platform', 'snippet_text']

present = [c for c in desired_cols if c in df.columns]

df = df[present].copy()

# Rename company to topic
df.rename(columns={"company": "topic"}, inplace=True)

print("Kept columns:", df.columns.tolist())

df.head()


# In[7]:


# Load score dates
schedule_file = "interview_schedule_2025.csv"
schedule_df = pd.read_csv(schedule_file)
print("Loaded:", schedule_file)
print("Shape:", schedule_df.shape)
schedule_df.head()


# In[9]:


# Convert date columns to datetimes (explicit formats: upload_date ISO, schedule day-month-year like 17-Dec-24)
df['upload_date'] = pd.to_datetime(df['upload_date'], format="%Y-%m-%dT%H:%M:%S", errors='coerce')
schedule_df['Start Date'] = pd.to_datetime(schedule_df['Start Date'], format="%d-%b-%y", errors='coerce')
schedule_df['End Date'] = pd.to_datetime(schedule_df['End Date'], format="%d-%b-%y", errors='coerce')

# Prepare Month/Year columns (nullable integers)
df['Month'] = pd.NA
df['Year'] = pd.NA

def _safe_month(val, fallback_dt):
    if pd.isna(val):
        return (pd.NA if pd.isna(fallback_dt) else int(fallback_dt.month))
    try:
        return int(val)
    except Exception:
        # try parsing textual month (e.g. "Dec" / "December")
        try:
            parsed = pd.to_datetime(str(val), errors='coerce')
            if pd.notna(parsed):
                return int(parsed.month)
        except Exception:
            pass
    return (pd.NA if pd.isna(fallback_dt) else int(fallback_dt.month))

def _safe_year(val, fallback_dt):
    if pd.isna(val):
        return (pd.NA if pd.isna(fallback_dt) else int(fallback_dt.year))
    try:
        return int(val)
    except Exception:
        try:
            parsed = pd.to_datetime(str(val), errors='coerce')
            if pd.notna(parsed):
                return int(parsed.year)
        except Exception:
            pass
    return (pd.NA if pd.isna(fallback_dt) else int(fallback_dt.year))

# For each schedule interval, assign Month and Year to matching upload_date rows
for _, srow in schedule_df[['Start Date', 'End Date', 'Month', 'Year']].dropna(subset=['Start Date', 'End Date']).iterrows():
    start, end = srow['Start Date'], srow['End Date']
    mon_raw, yr_raw = srow.get('Month'), srow.get('Year')
    mon = _safe_month(mon_raw, start)
    yr = _safe_year(yr_raw, start)
    mask = (df['upload_date'] >= start) & (df['upload_date'] <= end)
    df.loc[mask, 'Month'] = mon
    df.loc[mask, 'Year'] = yr

# convert to nullable integer dtypes for convenience
df['Month'] = df['Month'].astype('Int64')
df['Year'] = df['Year'].astype('Int64')

# Show result
print("Assigned months:", sorted(df['Month'].dropna().unique().tolist()))
df.head()


# In[11]:


# Remove rows with upload_date on or before 2024-12-16 (preserve rows with missing upload_date)
threshold = pd.to_datetime("2024-12-16")

mask_remove = df['upload_date'].notna() & (df['upload_date'] <= threshold)
removed_count = int(mask_remove.sum())

df = df.loc[~mask_remove].copy()

print(f"Removed {removed_count} rows with upload_date on or before {threshold.date()}")
print("New shape:", df.shape)
df.head()


# In[13]:


# Count missing values per column (count and percent)
missing_counts = df.isna().sum()
missing_pct = (df.isna().mean() * 100).round(2)

missing_summary = pd.DataFrame({
    "missing_count": missing_counts,
    "missing_pct": missing_pct
}).sort_values("missing_count", ascending=False)

print(missing_summary)


# In[15]:


# Drop any rows that contain missing values
before_count = len(df)
df = df.dropna().copy()
removed_count = before_count - len(df)

print(f"Removed {removed_count} rows with any missing values. New shape: {df.shape}")
df.head()

missing_counts = df.isna().sum()
missing_pct = (df.isna().mean() * 100).round(2)
missing_summary = pd.DataFrame({
    "missing_count": missing_counts,
    "missing_pct": missing_pct
}).sort_values("missing_count", ascending=False)
print(missing_summary)


# In[21]:


from transformers import pipeline, AutoTokenizer
import torch

HF_MODEL_PRIMARY   = "cardiffnlp/twitter-roberta-base-sentiment-latest"

model_name = HF_MODEL_PRIMARY if 'HF_MODEL_PRIMARY' in globals() else "cardiffnlp/twitter-roberta-base-sentiment-latest"
text_col = 'snippet_text'
if text_col not in df.columns:
    raise KeyError(f"Expected text column '{text_col}' not found in df")

device = 0 if torch.cuda.is_available() else -1

# load tokenizer with a safe max length and build pipeline with it
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
# enforce a reasonable max length (Roberta-based models typically 512)
tokenizer.model_max_length = min(getattr(tokenizer, "model_max_length", 512), 512)

sent_pipe = pipeline(
    "sentiment-analysis",
    model=model_name,
    tokenizer=tokenizer,
    device=device,
    return_all_scores=True
)

batch_size = 32
scores = []
n = len(df)

for i in range(0, n, batch_size):
    batch_texts = df[text_col].iloc[i:i+batch_size].fillna("").astype(str).tolist()
    # ensure inputs are truncated/padded to tokenizer.model_max_length to avoid oversized tensors
    results = sent_pipe(batch_texts, truncation=True, padding=True, max_length=tokenizer.model_max_length)
    for res in results:
        label_scores = {entry['label'].lower(): entry['score'] for entry in res}
        pos = label_scores.get('positive', 0.0)
        neg = label_scores.get('negative', 0.0)
        scores.append(pos - neg)

if len(scores) != n:
    raise RuntimeError(f"Score count ({len(scores)}) does not match df length ({n})")

df['sentiment'] = scores
print("Added 'sentiment' column to df (positive-negative score).")
df.head()
#


# In[107]:


import numpy as np

# Compute log(1 + count) and weight sentiment
df['sentiment_weighted'] = df['sentiment'] * np.log1p(df['impression_count_comb'])

# quick check
print(df[['sentiment', 'impression_count_comb', 'sentiment_weighted']].head())


# In[109]:


# Aggregate average sentiment_weighted for rows that have Month, Year, and topic
agg_df = (
    df.dropna(subset=['Month', 'Year', 'topic'])
      .groupby(['Year', 'Month', 'topic'], as_index=False)['sentiment_weighted']
      .mean()
      .rename(columns={'sentiment_weighted': 'sentiment_weighted_avg'})
)

# optional: sort for readability
agg_df = agg_df.sort_values(['Year', 'Month', 'topic']).reset_index(drop=True)

print("Aggregated shape:", agg_df.shape)
agg_df.head()


# In[111]:


# Pivot the aggregated data so each row is a Year-Month and each column is a topic with the weighted score
if 'agg_df' not in globals():
    raise NameError("Expected 'agg_df' to exist from previous cell (aggregated Year, Month, topic).")

wide_df = (
    agg_df
    .pivot_table(index=['Year', 'Month'], columns='topic', values='sentiment_weighted_avg', aggfunc='first')
    .sort_index()
)

# Convert Year/Month MultiIndex to a PeriodIndex (monthly) for easier time-based handling
try:
    wide_df.index = pd.PeriodIndex(
        year=wide_df.index.get_level_values('Year').astype(int),
        month=wide_df.index.get_level_values('Month').astype(int),
        freq='M'
    )
    wide_df.index.name = 'Period'
except Exception:
    # If conversion fails, keep the Year/Month MultiIndex (ensuring integer dtypes)
    wide_df = wide_df.rename_axis(index=['Year', 'Month'])

print("Wide dataframe shape:", wide_df.shape)
wide_df.head()


# In[113]:


# Load MCSI scores
scores_file = "scores.csv"
scores_df = pd.read_csv(scores_file)
print("Loaded:", scores_file)
print("Shape:", scores_df.shape)
scores_df.head()


# In[115]:


# Add MCSI scores to the main data frame
wide_df = wide_df.merge(
    scores_df,
    on=['Month', 'Year'],
    how='left'
)

wide_df
# Now we have one row for each month. That row has 10 features (an aggregated and weighted score for each topic) and an outcome variable (the MCSI score for that month)


# In[117]:


# Create lag feature (okay because we are splitting temporally for train/validate sets)
wide_df['prev_score'] = wide_df['score'].shift(1)

# Manually add back the score from 12/2024 to minimize data loss
wide_df.loc[(wide_df['Year'] == 2025) & (wide_df['Month'] == 1), 'prev_score'] = 74

wide_df


# In[119]:


# Aggregate to internal and external sentiment
internal_columns = ['Durable Goods and Big Purchases', 'Gasoline and Energy Prices', 'Income Expectations',
                    'Inflation and Prices','Personal Financial Situation','Unemployment and Job Security']
external_columns = ['Business and Economic Conditions', 'Government Policy and Inflation Control',
                    'Housing Market (Buying/Selling Homes)','Investments and Stock Market Confidence']
wide_df['internal_sentiment'] = wide_df[internal_columns].mean(axis=1)
wide_df['external_sentiment'] = wide_df[external_columns].mean(axis=1)

wide_df = wide_df.drop(['Durable Goods and Big Purchases', 'Gasoline and Energy Prices', 'Income Expectations',
                    'Inflation and Prices','Personal Financial Situation','Unemployment and Job Security',
                    'Business and Economic Conditions', 'Government Policy and Inflation Control',
                    'Housing Market (Buying/Selling Homes)','Investments and Stock Market Confidence'], axis=1)

wide_df


# In[121]:


# Create training and validation splits to fit models
train_df = wide_df[wide_df['Month'] < 9]
validate_df = wide_df[(wide_df['Month'] >= 9) & (wide_df['Month'] < 11)]


# In[123]:


# Splitting features and outcomes
# --- Prepare data ---
# Sort for readability
train_df = train_df.sort_values(by=["Year", "Month"]).reset_index(drop=True)
validate_df = validate_df.sort_values(by=["Year", "Month"]).reset_index(drop=True)

# Define features and target
exclude_cols = ["Month", "Year", "score"]
X_train = train_df.drop(columns=exclude_cols)
y_train = train_df["score"]

X_val = validate_df.drop(columns=exclude_cols)
y_val = validate_df["score"]


# In[125]:


# Ridge Model
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error

# --- Train Ridge model ---
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

# --- Predictions ---
train_df["predicted_score"] = ridge.predict(X_train)
validate_df["predicted_score"] = ridge.predict(X_val)

# --- Summary ---
print("=== Ridge Regression Summary ===")
print(f"Alpha (regularization strength): {ridge.alpha}")
print(f"R² (train): {ridge.score(X_train, y_train):.4f}")
print(f"R² (validation): {r2_score(y_val, validate_df['predicted_score']):.4f}")
import numpy as np
rmse_val = np.sqrt(mean_squared_error(y_val, validate_df['predicted_score']))
print(f"RMSE (validation): {rmse_val:.4f}")
print("\n--- Coefficients ---")
for name, coef in zip(X_train.columns, ridge.coef_):
    print(f"{name:20s} : {coef: .4f}")
print(f"\nIntercept: {ridge.intercept_:.4f}\n")

# --- Output Tables ---
print("=== Train Predictions (first 10) ===")
print(train_df[["Year", "Month", "score", "predicted_score"]].head(10))

print("\n=== Validation Predictions (first 10) ===")
print(validate_df[["Year", "Month", "score", "predicted_score"]].head(10))


# In[137]:


# --- Train Lasso model ---
lasso = Lasso(alpha=1.0, random_state=42)
lasso.fit(X_train, y_train)

# --- Predictions ---
train_df["predicted_score"] = lasso.predict(X_train)
validate_df["predicted_score"] = lasso.predict(X_val)

# --- Summary ---
print("=== Lasso Regression Summary ===")
print(f"Alpha (regularization strength): {lasso.alpha}")
print(f"R² (train): {lasso.score(X_train, y_train):.4f}")
print(f"R² (validation): {r2_score(y_val, validate_df['predicted_score']):.4f}")

rmse_val = np.sqrt(mean_squared_error(y_val, validate_df["predicted_score"]))
print(f"RMSE (validation): {rmse_val:.4f}")
print(f"\nIntercept: {lasso.intercept_:.4f}")

# --- Coefficients ---
print("\n=== Coefficients ===")
coeff_df = (
    pd.DataFrame({
        "Feature": X_train.columns,
        "Coefficient": lasso.coef_
    })
    .sort_values(by="Coefficient", ascending=False)
)
print(coeff_df)

# --- Output Tables ---
print("\n=== Train Predictions (first 10) ===")
print(train_df[["Year", "Month", "score", "predicted_score"]].head(10))

print("\n=== Validation Predictions (first 10) ===")
print(validate_df[["Year", "Month", "score", "predicted_score"]].head(10))


# In[139]:


import statsmodels.api as sm

# --- Prepare design matrices (add intercept) ---
X_train_ols = sm.add_constant(X_train, has_constant='add')
X_val_ols   = sm.add_constant(X_val,   has_constant='add')

# --- Fit OLS ---
ols_model = sm.OLS(y_train, X_train_ols).fit()

# --- Predictions ---
train_df["predicted_score"]    = ols_model.predict(X_train_ols)
validate_df["predicted_score"] = ols_model.predict(X_val_ols)

# --- Summary ---
print("=== OLS Regression Summary ===")
print(ols_model.summary())  # full table

# Also print quick metrics to mirror your Lasso block
print(f"\nR² (train): {ols_model.rsquared:.4f}")
r2_val   = r2_score(y_val, validate_df["predicted_score"])
rmse_val = np.sqrt(mean_squared_error(y_val, validate_df["predicted_score"]))
print(f"R² (validation): {r2_val:.4f}")
print(f"RMSE (validation): {rmse_val:.4f}")

# --- Coefficients ---
print("\n--- Coefficients ---")
# params is a Series indexed by ['const', <feature names>]
for name, coef in ols_model.params.items():
    print(f"{name:20s} : {coef: .4f}")

print(f"\nIntercept: {ols_model.params.get('const', np.nan):.4f}\n")

# --- Output Tables ---
print("=== Train Predictions (first 10) ===")
print(train_df[["Year", "Month", "score", "predicted_score"]].head(10))

print("\n=== Validation Predictions (first 10) ===")
print(validate_df[["Year", "Month", "score", "predicted_score"]].head(10))


# In[135]:


## Once we get more data, we might be able to expand to a variety of other models

## Then we pick a model and save the trained model to github and deploy it using a streamlit app. We can put all of the preprocessing 
## stuff into a function so that new data is cleaned as is appropriate and turned into a score. Explain exactly what data is going in with 
## The API stuff I did.

