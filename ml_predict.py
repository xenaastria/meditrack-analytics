import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# =========================================================
# 1. DATABASE CONNECTION
# =========================================================
DB_HOST = os.environ["DB_HOST"]
DB_USER = os.environ["DB_USER"]
DB_PASS = os.environ["DB_PASS"]
DB_NAME = os.environ["DB_NAME"]

engine = create_engine(
    f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}"
)

print("Connecting to database...")

# =========================================================
# 2. LOAD DISPENSE DATA
# =========================================================
try:
    df = pd.read_sql("SELECT * FROM dispense", engine)
except Exception as e:
    print("Failed to load dispense table:", e)
    sys.exit()

if df.empty:
    print("No dispense data found.")
    sys.exit()

print("Data loaded:", len(df))

# =========================================================
# 3. BASIC CLEANING
# =========================================================
required_cols = ["medicine_id", "quantity_dispensed", "dispense_date"]
missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    print("Missing required columns in dispense table:", missing_cols)
    sys.exit()

df = df.copy()
df["dispense_date"] = pd.to_datetime(df["dispense_date"], errors="coerce")
df["quantity_dispensed"] = pd.to_numeric(df["quantity_dispensed"], errors="coerce")

if "patient_age" in df.columns:
    df["patient_age"] = pd.to_numeric(df["patient_age"], errors="coerce")
else:
    df["patient_age"] = np.nan

df = df.dropna(subset=["medicine_id", "quantity_dispensed", "dispense_date"])

if df.empty:
    print("Dispense data became empty after cleaning.")
    sys.exit()

# =========================================================
# 4. CREATE CLIENT TYPE + AGE GROUP
# =========================================================
def get_age_group(age):
    if pd.isna(age):
        return "All"
    age = int(age)
    if 0 <= age <= 5:
        return "0-5"
    elif 6 <= age <= 12:
        return "6-12"
    elif 13 <= age <= 18:
        return "13-18"
    elif 19 <= age <= 59:
        return "19-59"
    else:
        return "60+"

def get_client_type(age):
    if pd.isna(age):
        return "All"
    age = int(age)
    return "Pedia" if age <= 18 else "Adult"

df["age_group"] = df["patient_age"].apply(get_age_group)
df["client_type"] = df["patient_age"].apply(get_client_type)

# =========================================================
# 5. AGGREGATE TO MONTHLY SEGMENTED DEMAND
# =========================================================
monthly = df.groupby(
    ["medicine_id", "client_type", "age_group", pd.Grouper(key="dispense_date", freq="M")]
)["quantity_dispensed"].sum().reset_index()

monthly = monthly.sort_values(
    ["medicine_id", "client_type", "age_group", "dispense_date"]
).reset_index(drop=True)

print("\nMonthly segmented data:")
print(monthly.head(20))

if monthly.empty:
    print("No monthly aggregated data found.")
    sys.exit()

# =========================================================
# 6. CREATE / UPDATE FORECAST TABLE
# =========================================================
create_table_sql = """
CREATE TABLE IF NOT EXISTS ml_forecast (
    id INT AUTO_INCREMENT PRIMARY KEY,
    medicine_id INT NOT NULL,
    client_type VARCHAR(20) NOT NULL,
    age_group VARCHAR(20) NOT NULL,
    forecast_month DATE NULL,
    predicted_quantity FLOAT DEFAULT 0,
    predicted_next_month FLOAT DEFAULT 0,
    predicted_3_months FLOAT DEFAULT 0,
    predicted_6_months FLOAT DEFAULT 0,
    predicted_12_months FLOAT DEFAULT 0,
    predicted_24_months FLOAT DEFAULT 0,
    predicted_36_months FLOAT DEFAULT 0,
    model_mae FLOAT DEFAULT 0,
    model_type VARCHAR(50) DEFAULT 'fallback',
    generated_at DATETIME
)
"""

with engine.begin() as conn:
    conn.execute(text(create_table_sql))

    try:
        conn.execute(text("ALTER TABLE ml_forecast ADD COLUMN forecast_month DATE NULL"))
    except Exception:
        pass

    try:
        conn.execute(text("ALTER TABLE ml_forecast ADD COLUMN predicted_quantity FLOAT DEFAULT 0"))
    except Exception:
        pass

# =========================================================
# 7. COUNT MONTHLY RECORDS PER SEGMENT
# =========================================================
segment_counts = monthly.groupby(
    ["medicine_id", "client_type", "age_group"]
).size().reset_index(name="month_count")

print("\nMonthly record count per segment:")
print(segment_counts)

eligible_segments = segment_counts[segment_counts["month_count"] >= 4].copy()

# =========================================================
# 8. HELPER: CALCULATE LONG-TERM PROJECTIONS
# =========================================================
def build_projection_columns(frame, next_col="predicted_next_month", trend_col=None):
    result = frame.copy()

    if trend_col is None or trend_col not in result.columns:
        result["predicted_3_months"] = result[next_col] * 3
        result["predicted_6_months"] = result[next_col] * 6
        result["predicted_12_months"] = result[next_col] * 12
        result["predicted_24_months"] = result[next_col] * 24
        result["predicted_36_months"] = result[next_col] * 36
    else:
        capped_trend = result[trend_col].clip(lower=-0.20, upper=0.20)

        result["predicted_3_months"] = result[next_col] * 3 * (1 + capped_trend)
        result["predicted_6_months"] = result[next_col] * 6 * (1 + capped_trend)
        result["predicted_12_months"] = result[next_col] * 12 * (1 + capped_trend)
        result["predicted_24_months"] = result[next_col] * 24 * ((1 + capped_trend) ** 2)
        result["predicted_36_months"] = result[next_col] * 36 * ((1 + capped_trend) ** 3)

    proj_cols = [
        "predicted_3_months",
        "predicted_6_months",
        "predicted_12_months",
        "predicted_24_months",
        "predicted_36_months"
    ]

    result[next_col] = result[next_col].clip(lower=0)
    for col in proj_cols:
        result[col] = result[col].clip(lower=0)

    return result

# =========================================================
# 9. FALLBACK FORECAST FOR ALL SEGMENTS
# =========================================================
def build_fallback_forecast(monthly_df):
    fallback_rows = []

    for keys, segment in monthly_df.groupby(["medicine_id", "client_type", "age_group"]):
        segment = segment.sort_values("dispense_date").copy()

        med_id, client_type, age_group = keys
        avg_monthly = float(segment["quantity_dispensed"].mean())

        last_date = segment["dispense_date"].max()
        forecast_month = (last_date + pd.offsets.MonthBegin(1)).date()

        if len(segment) >= 2:
            first_val = float(segment["quantity_dispensed"].iloc[0])
            last_val = float(segment["quantity_dispensed"].iloc[-1])
            trend_rate = ((last_val - first_val) / first_val) if first_val > 0 else 0.0
        else:
            trend_rate = 0.0

        next_month_value = max(0.0, avg_monthly)

        fallback_rows.append({
            "medicine_id": int(med_id),
            "client_type": str(client_type),
            "age_group": str(age_group),
            "forecast_month": forecast_month,
            "predicted_quantity": next_month_value,
            "predicted_next_month": next_month_value,
            "trend_rate": float(trend_rate),
            "model_mae": 0.0,
            "model_type": "average_fallback",
            "generated_at": datetime.now()
        })

    fallback_df = pd.DataFrame(fallback_rows)

    if fallback_df.empty:
        return fallback_df

    fallback_df = build_projection_columns(
        fallback_df,
        next_col="predicted_next_month",
        trend_col="trend_rate"
    )

    return fallback_df[
        [
            "medicine_id",
            "client_type",
            "age_group",
            "forecast_month",
            "predicted_quantity",
            "predicted_next_month",
            "predicted_3_months",
            "predicted_6_months",
            "predicted_12_months",
            "predicted_24_months",
            "predicted_36_months",
            "model_mae",
            "model_type",
            "generated_at"
        ]
    ]

fallback_all = build_fallback_forecast(monthly)

# =========================================================
# 10. SAVE TO DATABASE
# =========================================================
def save_forecast_to_db(result_df):
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM ml_forecast"))

        insert_sql = text("""
            INSERT INTO ml_forecast (
                medicine_id,
                client_type,
                age_group,
                forecast_month,
                predicted_quantity,
                predicted_next_month,
                predicted_3_months,
                predicted_6_months,
                predicted_12_months,
                predicted_24_months,
                predicted_36_months,
                model_mae,
                model_type,
                generated_at
            ) VALUES (
                :medicine_id,
                :client_type,
                :age_group,
                :forecast_month,
                :predicted_quantity,
                :predicted_next_month,
                :predicted_3_months,
                :predicted_6_months,
                :predicted_12_months,
                :predicted_24_months,
                :predicted_36_months,
                :model_mae,
                :model_type,
                :generated_at
            )
        """)

        records = result_df.to_dict(orient="records")
        conn.execute(insert_sql, records)

if eligible_segments.empty:
    print("\nNot enough monthly data for ML in any segment. Using fallback only.")
    save_forecast_to_db(fallback_all)
    print("\nFallback forecast saved successfully.")
    print(fallback_all.head(20))
    sys.exit()

# =========================================================
# 11. PREPARE ONLY ELIGIBLE SEGMENTS FOR ML
# =========================================================
ml_data = monthly.merge(
    eligible_segments[["medicine_id", "client_type", "age_group"]],
    on=["medicine_id", "client_type", "age_group"],
    how="inner"
)

def create_features(segment):
    segment = segment.sort_values("dispense_date").copy()

    segment["lag_1"] = segment["quantity_dispensed"].shift(1)
    segment["lag_2"] = segment["quantity_dispensed"].shift(2)
    segment["lag_3"] = segment["quantity_dispensed"].shift(3)
    segment["moving_avg"] = segment["quantity_dispensed"].rolling(window=3).mean()

    segment["month_num"] = segment["dispense_date"].dt.month
    segment["year_num"] = segment["dispense_date"].dt.year

    segment["trend_rate"] = np.where(
        segment["lag_1"] > 0,
        (segment["quantity_dispensed"] - segment["lag_1"]) / segment["lag_1"],
        0.0
    )

    return segment

ml_data = ml_data.groupby(
    ["medicine_id", "client_type", "age_group"],
    group_keys=False
).apply(create_features)

ml_data = ml_data.dropna(subset=[
    "lag_1", "lag_2", "lag_3", "moving_avg", "month_num", "year_num", "quantity_dispensed"
]).reset_index(drop=True)

print("\nFeature-ready ML data:")
print(ml_data.head(20))

if ml_data.empty:
    print("\nFeature-ready ML dataset is empty. Saving fallback forecasts only.")
    save_forecast_to_db(fallback_all)
    print("\nFallback forecast saved successfully.")
    sys.exit()

# =========================================================
# 12. TRAIN MODEL
# =========================================================
feature_cols = ["lag_1", "lag_2", "lag_3", "moving_avg", "month_num", "year_num"]

X = ml_data[feature_cols]
y = ml_data["quantity_dispensed"]

if len(X) < 2:
    print("\nNot enough feature rows for ML training. Saving fallback forecasts only.")
    save_forecast_to_db(fallback_all)
    print("\nFallback forecast saved successfully.")
    sys.exit()

split = int(len(X) * 0.8)
if split < 1:
    split = 1
if split >= len(X):
    split = len(X) - 1

X_train = X.iloc[:split]
X_test = X.iloc[split:]
y_train = y.iloc[:split]
y_test = y.iloc[split:]

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)
model.fit(X_train, y_train)

preds = model.predict(X_test)
mae = float(mean_absolute_error(y_test, preds)) if len(y_test) > 0 else 0.0

print("\nModel trained successfully.")
print("MAE:", mae)

# =========================================================
# 13. PREDICT NEXT MONTH FOR ELIGIBLE SEGMENTS
# =========================================================
latest = ml_data.groupby(
    ["medicine_id", "client_type", "age_group"],
    group_keys=False
).tail(1).copy()

future_X = latest[feature_cols].copy()

future_X["month_num"] = future_X["month_num"] + 1
december_mask = future_X["month_num"] > 12
future_X.loc[december_mask, "month_num"] = 1
future_X.loc[december_mask, "year_num"] = future_X.loc[december_mask, "year_num"] + 1

next_month_pred = model.predict(future_X)
next_month_pred = np.clip(next_month_pred, a_min=0, a_max=None)

forecast_months = []
for _, row in latest.iterrows():
    year_num = int(row["year_num"])
    month_num = int(row["month_num"]) + 1
    if month_num > 12:
        month_num = 1
        year_num += 1
    forecast_months.append(pd.Timestamp(year=year_num, month=month_num, day=1).date())

ml_result = latest[
    ["medicine_id", "client_type", "age_group", "trend_rate"]
].copy()

ml_result["forecast_month"] = forecast_months
ml_result["predicted_quantity"] = next_month_pred
ml_result["predicted_next_month"] = next_month_pred
ml_result["model_mae"] = mae
ml_result["model_type"] = "random_forest"
ml_result["generated_at"] = datetime.now()

ml_result = build_projection_columns(
    ml_result,
    next_col="predicted_next_month",
    trend_col="trend_rate"
)

ml_result = ml_result[
    [
        "medicine_id",
        "client_type",
        "age_group",
        "forecast_month",
        "predicted_quantity",
        "predicted_next_month",
        "predicted_3_months",
        "predicted_6_months",
        "predicted_12_months",
        "predicted_24_months",
        "predicted_36_months",
        "model_mae",
        "model_type",
        "generated_at"
    ]
]

# =========================================================
# 14. MERGE ML RESULTS + FALLBACK RESULTS
# =========================================================
ml_keys = ml_result[["medicine_id", "client_type", "age_group"]].copy()

fallback_remaining = fallback_all.merge(
    ml_keys,
    on=["medicine_id", "client_type", "age_group"],
    how="left",
    indicator=True
)

fallback_remaining = fallback_remaining[
    fallback_remaining["_merge"] == "left_only"
].drop(columns=["_merge"])

final_result = pd.concat([ml_result, fallback_remaining], ignore_index=True)

final_result = final_result.sort_values(
    ["medicine_id", "client_type", "age_group"]
).reset_index(drop=True)

# =========================================================
# 15. SAVE TO DATABASE
# =========================================================
save_forecast_to_db(final_result)

print("\nForecast saved to ml_forecast successfully.")
print(final_result.head(30))

# =========================================================
# 16. OPTIONAL SUMMARY
# =========================================================
print("\nSummary:")
print("Total forecast rows:", len(final_result))
print("ML rows:", len(ml_result))
print("Fallback rows:", len(fallback_remaining))
print("Done.")