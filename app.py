import streamlit as st
import os
import statsmodels.api as sm


# --- Load environment variables securely ---
auth_endpoint = os.getenv("AUTH_ENDPOINT")
api_endpoint = os.getenv("API_ENDPOINT")
client_id = os.getenv("CLIENT_ID")
client_secret = os.getenv("CLIENT_SECRET")
scope = os.getenv("SCOPE")
api_key = os.getenv("API_KEY")

import json, requests
from requests.auth import HTTPBasicAuth

data_points = st.slider(label = "How many snippets per topic?",
						min_value = 1,
						max_value = 3000,
						value = 5)


from model_utils import ridge

def load_ols():
    ols = joblib.load("ols.joblib")
    return ols

ols_model = load_ols

## Workflow
if st.button("Refresh"):
    with st.spinner("Gathering Data ⏳"):
        def get_access_token():
            """Fetch OAuth token using client credentials."""
            r = requests.post(
                auth_endpoint,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                auth=HTTPBasicAuth(client_id.strip(), client_secret.strip()),
                data={"grant_type": "client_credentials", "scope": scope},
                timeout=20,
            )
            print("[AUTH]", r.status_code)
            r.raise_for_status()
            token = r.json().get("access_token")
            if not token:
                raise RuntimeError("No access_token in response.")
            return token
        st.markdown("1. ✅ Authenticated!")

        def call_filtered(q_filters=None, keywords=None, result_count = 1, label=None):


            token = get_access_token()

            headers = {
                "Authorization": token,  # or f"Bearer {token}" if required
                "x-api-key": api_key,
                "Content-Type": "application/json"
            }

            params = {
                "q": q_filters,
                "sort": "impression_count:desc",
                "per_page": result_count,
                "page": 1
            }

            payload = {
                "conditions": [
                    {"keywords_expression": keywords}
                ]
            }

            print("➡️ Query params:", params)
            print("➡️ Body:", payload)

            r = requests.post(api_endpoint, headers=headers, params=params,
                            data=json.dumps(payload), timeout=30)
            print("[EXPLORE] status:", r.status_code)

            data = r.json()
            print(json.dumps({k: data.get(k) for k in ("total_count", "items")}, indent=2))

            items = data.get("items", [])
            st.markdown(f"\n✅ {len(items)} items found in {label}")
            for i, it in enumerate(items[:5], 1):
                print(f"{i}. [{it.get('platform_name')}] {it.get('post_title')}")
                print((it.get('snippet_text') or '')[:200], "\n")

            return data

        booleans = [
            {
                "name": "Personal_Financial_Situation",
                "query": '("personal finances" OR "financial situation" OR "better off" OR "worse off" OR "making ends meet" OR "money problems" OR "cost of living" OR "household budget")'
            },
            {
                "name": "Business_and_Economic_Conditions",
                "query": '("economy" OR "economic outlook" OR "recession" OR "economic growth" OR "business conditions" OR "economic slowdown" OR "unemployment" OR "economic recovery")'
            },
            {
                "name": "Government_Policy_and_Inflation_Control",
                "query": '("government policy" OR "economic policy" OR "inflation control" OR "stimulus" OR "interest rates" OR "job creation" OR "federal reserve" OR "monetary policy")'
            },
            {
                "name": "Unemployment_and_Job_Security",
                "query": '("unemployment" OR "layoffs" OR "job loss" OR "hiring freeze" OR "job security" OR "labor market" OR "getting a job" OR "fired" OR "lost my job")'
            },
            {
                "name": "Inflation_and_Prices",
                "query": '("inflation" OR "price increase" OR "cost of goods" OR "rising prices" OR "cost of living" OR "prices going up" OR "deflation" OR "price stability")'
            },
            {
                "name": "Income_Expectations",
                "query": '("income growth" OR "wage increase" OR "salary raise" OR "higher pay" OR "income expectations" OR "earning potential" OR "financial improvement")'
            },
            {
                "name": "Housing_Market",
                "query": '("housing market" OR "home prices" OR "buying a house" OR "selling a house" OR "real estate market" OR "mortgage rates" OR "housing affordability")'
            },
            {
                "name": "Durable_Goods_and_Big_Purchases",
                "query": '("consumer spending" OR "buying a car" OR "buying furniture" OR "big purchases" OR "appliance sales" OR "vehicle market" OR "consumer confidence")'
            },
            {
                "name": "Gasoline_and_Energy_Prices",
                "query": '("gas prices" OR "fuel costs" OR "energy prices" OR "price of gasoline" OR "oil prices" OR "gasoline inflation" OR "cost of fuel")'
            },
            {
                "name": "Investments_and_Stock_Market_Confidence",
                "query": '("stock market" OR "mutual funds" OR "investments" OR "401k" OR "retirement savings" OR "market volatility" OR "financial markets" OR "stock prices")'
            }
        ]

        from datetime import datetime, timedelta

        # === Dynamic date window ===
        today = datetime.today().date()
        one_month_ago = today - timedelta(days=30)

        # Format as ISO 8601 strings (YYYY-MM-DD)
        start_date = one_month_ago.isoformat()
        end_date = today.isoformat()

        # Build q_string dynamically
        q_string = f"start_date:gte:{start_date},end_date:lte:{end_date}"

        st.markdown(f"2. ✅ Set the date range for past 30 days: {q_string}")


        ## This pulls the 3,000 most-viewed snippets from each of our topics over the last month. Then is randomly samples 300 from the top 3,000
        ## in each topic. With 10 topics, this leaves us with 3,000 snippets to run through our model and get a live consumer sentiment index score
        ## based on the last 30 days of data.

        import csv
        import random

        desired_results = data_points
        csv_filename = "pendulum_results.csv"

        fieldnames = [
            "snippet_text",
            "upload_date",
            "impression_count_comb",
            "post_link",
            "platform",   
            "post_title",
            "post_description",
            "creator_name",
            "creator_link",
            "creator_total_followers",
            "company", 
            "query_used",              
        ]

        all_rows = []
        st.markdown("3. ✅ Data Loading:")
        for q in booleans:
            query_name = q["name"]
            keywords_string = q["query"]

            print(f"🔍 Running query '{query_name}' ...")

            data = call_filtered(q_filters=q_string, keywords=keywords_string, result_count=desired_results, label=query_name)
            items = data.get("items", [])

            if not items:
                print(f"⚠️ No items returned for: {query_name}")
                continue

            sample_size = min(300, len(items))
            sampled_items = random.sample(items, sample_size)

            for item in sampled_items:
                row = {
                    "snippet_text": item.get("snippet_text", ""),
                    "upload_date": item.get("snippet_posted_datetime", ""),   
                    "impression_count_comb": item.get("total_views", ""),     
                    "post_link": item.get("post_link", ""),
                    "platform": item.get("platform_name", ""),    
                    "post_title": item.get("post_title", ""),
                    "post_description": item.get("post_description", ""),
                    "creator_name": item.get("creator_name", ""),
                    "creator_link": item.get("creator_link", ""),
                    "creator_total_followers": item.get("creator_total_followers", ""),
                    "company": query_name,
                    "query_used": keywords_string,
                }
                all_rows.append(row)

        # ---- Save all combined results to CSV ----
        if all_rows:
            with open(csv_filename, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_rows)

            st.markdown(f"💾 Loaded {len(all_rows)} total sampled records across {len(booleans)} topics")
        else:
            st.markdown("⚠️ No data to save.")

        # Load in the data
        import pandas as pd

        file = "pendulum_results.csv"

        df = pd.read_csv(file)
        print("Loaded:", file)
        print("Shape:", df.shape)
        df.head()

        # Keep only the necessary columns
        desired_cols = ['company', 'upload_date', 'impression_count_comb', 'platform', 'snippet_text']

        present = [c for c in desired_cols if c in df.columns]

        df = df[present].copy()

        # Rename company to topic
        df.rename(columns={"company": "topic"}, inplace=True)

        print("Kept columns:", df.columns.tolist())

        df.head()

        # Count missing values per column (count and percent)
        missing_counts = df.isna().sum()
        missing_pct = (df.isna().mean() * 100).round(2)

        missing_summary = pd.DataFrame({
            "missing_count": missing_counts,
            "missing_pct": missing_pct
        }).sort_values("missing_count", ascending=False)

        print(missing_summary)

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
        st.markdown("4. ✅ Input data cleaned.")


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
        st.markdown("5. ✅ Added 'sentiment' column to df (positive-negative score).")

        import numpy as np

        # Compute log(1 + count) and weight sentiment
        df['sentiment_weighted'] = df['sentiment'] * np.log1p(df['impression_count_comb'])
        st.markdown("6. ✅ Sentiment scores weighted")


        # Aggregate average sentiment_weighted by topic
        agg_df = (
            df.dropna(subset=['topic'])
            .groupby('topic', as_index=False)['sentiment_weighted']
            .mean()
            .rename(columns={'sentiment_weighted': 'sentiment_weighted_avg'})
        )
        st.markdown("7. ✅ Aggregated by month")

        # Pivoting the dataframe to run through model
        wide_df = (
            agg_df
            .pivot_table( columns='topic', values='sentiment_weighted_avg', aggfunc='first')
            .sort_index()
        )

        st.markdown("8. ✅ Table pivoted for model")

        # Aggregating by internal and external sentiment
        internal_columns = ['Durable_Goods_and_Big_Purchases', 'Gasoline_and_Energy_Prices', 'Income_Expectations',
                            'Inflation_and_Prices','Personal_Financial_Situation','Unemployment_and_Job_Security']
        external_columns = ['Business_and_Economic_Conditions', 'Government_Policy_and_Inflation_Control',
                            'Housing_Market','Investments_and_Stock_Market_Confidence']
        wide_df['internal_sentiment'] = wide_df[internal_columns].mean(axis=1)
        wide_df['external_sentiment'] = wide_df[external_columns].mean(axis=1)

        wide_df = wide_df.drop(['Durable_Goods_and_Big_Purchases', 'Gasoline_and_Energy_Prices', 'Income_Expectations',
                            'Inflation_and_Prices','Personal_Financial_Situation','Unemployment_and_Job_Security',
                            'Business_and_Economic_Conditions', 'Government_Policy_and_Inflation_Control',
                            'Housing_Market','Investments_and_Stock_Market_Confidence'], axis=1)
        st.markdown("9. ✅ Data aggregated to internal and external sentiment scores")

        # Manually adding in the score from the prior month
        wide_df['prev_score'] = 53.6
    st.success("Data is prepared for the model ✅")
    st.write(wide_df)
    with st.spinner("Scoring data ⏳"):
        ridge_outcome = ridge(wide_df)
        lasso_outcome = lasso.predict(wide_df)
        ols_outcome = ols_model.predict(wide_df)
    st.success("Analysis complete")
    st.markdown(f"## The current consumer sentiment score is {ols_outcome}, {lasso_outcome}, {ridge_outcome}")