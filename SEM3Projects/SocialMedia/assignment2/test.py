# E-Commerce Clickstream Data Analytics

## Setup and Imports

import datetime
import warnings
from decimal import Decimal

import matplotlib.pyplot as plt
import numpy as np

# Import necessary libraries
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore", category=FutureWarning)

# More modern imports for machine learning models
import lightgbm as lgb

# Deep learning imports (using PyTorch)
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Define file paths
clicks_file = "data/yoochoose-data/yoochoose-clicks.dat"
buys_file = "data/yoochoose-data/yoochoose-buys.dat"

# Setting up better visualization defaults
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12

# Sampling configuration - Option 1
SAMPLE_MODE = True  # Set to False to process the full dataset
MAX_CHUNKS = 2  # Process only first 2 chunks instead of all 67
SAMPLE_SESSIONS = 1000  # Limit to this many sessions

print("Setup complete!")
print(
    f"Sampling mode: {'ON - Using a subset of data for testing' if SAMPLE_MODE else 'OFF - Using full dataset'}"
)

if SAMPLE_MODE:
    print(
        f"Processing max {MAX_CHUNKS} chunks with up to {SAMPLE_SESSIONS} sessions per chunk"
    )

## Helper Functions


def convert_category(x):
    """
    Handle Category column in the clicks data (Numerical Encoding 1-12 for Category of Items)
    """
    if x == "S":
        return -1
    elif x in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"]:
        return x
    else:
        return 13


def check_item_in_purchases(first_item, last_item, unique_items_bought, is_buy):
    """
    Check if first or last clicked item is in the set of purchased items
    """
    if is_buy == 1:
        if first_item in unique_items_bought or last_item in unique_items_bought:
            return True
    return False


def get_preds(threshold, probabilities):
    """
    Convert probabilities to binary predictions based on threshold
    """
    return [1 if prob > threshold else 0 for prob in probabilities]


def calc_special_offer_click(categories):
    """
    Determine if a session included a special offer click
    """
    if -1 in categories:
        return 1
    else:
        return 0


def p_root(value, root):
    """
    Calculate p-root of a value
    """
    root_value = 1 / float(root)
    return round(Decimal(value) ** Decimal(root_value), 3)


def minkowski_distance(x, y, p_value):
    """
    Calculate Minkowski distance between two vectors
    """
    return p_root(sum(pow(abs(a - b), p_value) for a, b in zip(x, y)), p_value)


## Data Loading

print("Loading buys data...")
# Load buys data
buys = pd.read_csv(buys_file, names=["session", "timestamp", "item_id", "price", "qty"])

# Data Loading for Clicks (processing in chunks will be done later)
print("Setting up columns for clicks data...")
clicks_columns = ["session", "timestamp", "item_id", "category"]

# Display initial output to confirm data loading
print(f"Buys data shape: {buys.shape}")
print(buys.head())

## Data Exploration - Buys

print("Exploring buys data...")

# Top 10 Items which have been bought the maximum
plt.figure(figsize=(14, 7))
top_bought_items = buys["item_id"].value_counts().nlargest(10)
sns.barplot(x=top_bought_items.index, y=top_bought_items.values)
plt.title("Top 10 Items with Maximum Purchases")
plt.xlabel("Item ID")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Top 10 items which are purchased in larger quantities
quantity_analysis = (
    buys.groupby("item_id")["qty"].sum().sort_values(ascending=False).nlargest(10)
)
plt.figure(figsize=(14, 7))
sns.barplot(x=quantity_analysis.index, y=quantity_analysis.values)
plt.title("Top 10 Items Purchased in Larger Quantities")
plt.xlabel("Item ID")
plt.ylabel("Total Quantity")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Top 10 items with maximum price
price_analysis = (
    buys[["item_id", "price"]]
    .drop_duplicates()
    .sort_values("price", ascending=False)
    .nlargest(10, "price")
)
plt.figure(figsize=(14, 7))
sns.barplot(x=price_analysis["item_id"], y=price_analysis["price"])
plt.title("Top 10 Items with Highest Prices")
plt.xlabel("Item ID")
plt.ylabel("Price")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

## Transforming Buys Data

print("Transforming buys data...")
# Group buys by session
grouped_buys = buys.groupby("session")
buys_transformed = pd.DataFrame(index=grouped_buys.groups.keys())

# Calculate metrics for each buying session
buys_transformed["Number_items_bought"] = grouped_buys.item_id.count()
buys_transformed["unique_items_bought"] = grouped_buys.item_id.apply(
    lambda x: list(x.unique())
)
buys_transformed["is_buy"] = 1
buys_transformed.index.name = "session"

print("Buys transformation complete")
print(buys_transformed.head())

## Loading and Transforming Clicks Data (in chunks)

# Step 1: Process clicks for session timing analysis
print("Processing clicks data in chunks (session timing analysis)...")
result_timing = None
count = 1

for chunk in pd.read_csv(
    clicks_file,
    names=clicks_columns,
    usecols=["session", "timestamp", "item_id", "category"],
    parse_dates=["timestamp"],
    chunksize=500000,
):
    print(f"Processing chunk {count} for timing analysis...")

    # SAMPLING: If in sample mode, take only a subset of sessions
    if SAMPLE_MODE:
        unique_sessions = chunk["session"].unique()
        if len(unique_sessions) > SAMPLE_SESSIONS:
            sample_sessions = np.random.choice(
                unique_sessions, SAMPLE_SESSIONS, replace=False
            )
            chunk = chunk[chunk["session"].isin(sample_sessions)]

    # Group by session to analyze timing
    chunk_grouped = chunk.groupby("session")
    chunk_timing = pd.DataFrame()

    # Calculate session timing metrics
    chunk_timing["min"] = chunk_grouped["timestamp"].min()
    chunk_timing["max"] = chunk_grouped["timestamp"].max()
    chunk_timing["dwell_time"] = chunk_timing["max"] - chunk_timing["min"]
    chunk_timing["dwell_time_seconds"] = chunk_timing["dwell_time"].dt.total_seconds()
    chunk_timing["total_clicks"] = chunk_grouped.size()
    chunk_timing["dayofweek"] = chunk_timing["min"].dt.dayofweek
    chunk_timing["dayofmonth"] = chunk_timing["min"].dt.day
    chunk_timing["hourofclick"] = chunk_timing["min"].dt.hour

    # Create time of day categories
    time_bins = [0, 4, 8, 12, 16, 20, 24]
    time_labels = ["Late Night", "Early Morning", "Morning", "Noon", "Evening", "Night"]
    chunk_timing["timeofday"] = pd.cut(
        chunk_timing["hourofclick"],
        bins=time_bins,
        labels=time_labels,
        include_lowest=True,
    )

    # Calculate click rate (clicks per second)
    chunk_timing["click_rate"] = (
        chunk_timing["total_clicks"] / chunk_timing["dwell_time_seconds"]
    )
    chunk_timing["click_rate"] = chunk_timing["click_rate"].replace(np.inf, np.nan)
    chunk_timing["click_rate"] = chunk_timing["click_rate"].fillna(0)

    # Combine results
    if result_timing is None:
        result_timing = chunk_timing
    else:
        result_timing = pd.concat([result_timing, chunk_timing])

    count += 1

    # SAMPLING: Stop after MAX_CHUNKS in sample mode
    if SAMPLE_MODE and count > MAX_CHUNKS:
        print(f"Stopping after {MAX_CHUNKS} chunks (sample mode)")
        break

print("Completed timing analysis of clicks data")

# Step 2: Process clicks for item analysis
print("Processing clicks data in chunks (item analysis)...")
result_items = None
count = 1

for chunk in pd.read_csv(
    clicks_file,
    names=clicks_columns,
    usecols=["session", "item_id", "category"],
    chunksize=500000,
):
    print(f"Processing chunk {count} for item analysis...")

    # SAMPLING: If in sample mode, take only a subset of sessions
    if SAMPLE_MODE:
        unique_sessions = chunk["session"].unique()
        if len(unique_sessions) > SAMPLE_SESSIONS:
            sample_sessions = np.random.choice(
                unique_sessions, SAMPLE_SESSIONS, replace=False
            )
            chunk = chunk[chunk["session"].isin(sample_sessions)]

    # Group by session
    chunk_grouped = chunk.groupby("session")

    # Get first/last clicked items and unique counts
    chunk_items = pd.DataFrame()
    chunk_items["first_clicked_item"] = chunk_grouped["item_id"].first()
    chunk_items["last_clicked_item"] = chunk_grouped["item_id"].last()
    chunk_items["total_unique_items"] = chunk_grouped["item_id"].nunique()
    chunk_items["total_unique_categories"] = chunk_grouped["category"].nunique()

    # Combine results
    if result_items is None:
        result_items = chunk_items
    else:
        result_items = pd.concat([result_items, chunk_items])

    count += 1

    # SAMPLING: Stop after MAX_CHUNKS in sample mode
    if SAMPLE_MODE and count > MAX_CHUNKS:
        print(f"Stopping after {MAX_CHUNKS} chunks (sample mode)")
        break

print("Completed item analysis of clicks data")

# Step 3: Process clicks for sequence analysis
print("Processing clicks data in chunks (sequence analysis)...")
visited_items_by_session = {}
count = 1

for chunk in pd.read_csv(
    clicks_file, names=clicks_columns, usecols=["session", "item_id"], chunksize=500000
):
    print(f"Processing chunk {count} for sequence analysis...")

    # SAMPLING: If in sample mode, take only a subset of sessions
    if SAMPLE_MODE:
        unique_sessions = chunk["session"].unique()
        if len(unique_sessions) > SAMPLE_SESSIONS:
            sample_sessions = np.random.choice(
                unique_sessions, SAMPLE_SESSIONS, replace=False
            )
            chunk = chunk[chunk["session"].isin(sample_sessions)]

    # Group items by session
    for session, items in chunk.groupby("session")["item_id"]:
        if session in visited_items_by_session:
            visited_items_by_session[session].extend(items.tolist())
        else:
            visited_items_by_session[session] = items.tolist()

    count += 1

    # SAMPLING: Stop after MAX_CHUNKS in sample mode
    if SAMPLE_MODE and count > MAX_CHUNKS:
        print(f"Stopping after {MAX_CHUNKS} chunks (sample mode)")
        break

# Convert to DataFrame
result_sequences = pd.DataFrame(
    {
        "session": list(visited_items_by_session.keys()),
        "visited_items": list(visited_items_by_session.values()),
    }
).set_index("session")

print("Completed sequence analysis of clicks data")

# Step 4: Process clicks for category analysis
print("Processing clicks data in chunks (category analysis)...")
visited_categories_by_session = {}
count = 1

for chunk in pd.read_csv(
    clicks_file,
    names=clicks_columns,
    usecols=["session", "category"],
    converters={"category": convert_category},
    chunksize=500000,
):
    print(f"Processing chunk {count} for category analysis...")

    # SAMPLING: If in sample mode, take only a subset of sessions
    if SAMPLE_MODE:
        unique_sessions = chunk["session"].unique()
        if len(unique_sessions) > SAMPLE_SESSIONS:
            sample_sessions = np.random.choice(
                unique_sessions, SAMPLE_SESSIONS, replace=False
            )
            chunk = chunk[chunk["session"].isin(sample_sessions)]

    # Group categories by session
    for session, categories in chunk.groupby("session")["category"]:
        if session in visited_categories_by_session:
            visited_categories_by_session[session].extend(categories.tolist())
        else:
            visited_categories_by_session[session] = categories.tolist()

    count += 1

    # SAMPLING: Stop after MAX_CHUNKS in sample mode
    if SAMPLE_MODE and count > MAX_CHUNKS:
        print(f"Stopping after {MAX_CHUNKS} chunks (sample mode)")
        break

# Convert to DataFrame
result_categories = pd.DataFrame(
    {
        "session": list(visited_categories_by_session.keys()),
        "visited_categories": list(visited_categories_by_session.values()),
    }
).set_index("session")

# Add derived metrics
result_categories["Number_clicked_visited_categories"] = result_categories[
    "visited_categories"
].apply(len)
result_categories["Special_offer_click"] = result_categories[
    "visited_categories"
].apply(calc_special_offer_click)

print("Completed category analysis of clicks data")

## Calculate Item Popularity

print("Calculating item popularity from clicks...")
# Clicks popularity
clicks_counts = pd.DataFrame()
count = 1

for chunk in pd.read_csv(
    clicks_file, names=clicks_columns, usecols=["item_id"], chunksize=500000
):
    print(f"Processing chunk {count} for item popularity...")

    # SAMPLING: If in sample mode, we still want to process more chunks for popularity
    # to get a more representative distribution, but we can cap it
    if SAMPLE_MODE and count > MAX_CHUNKS * 2:
        print(
            f"Stopping popularity calculation after {MAX_CHUNKS * 2} chunks (sample mode)"
        )
        break

    item_counts = chunk["item_id"].value_counts()

    if clicks_counts.empty:
        clicks_counts = pd.DataFrame(item_counts)
        clicks_counts.columns = ["count"]
    else:
        # Add counts to existing items or create new items
        tmp_counts = pd.DataFrame(item_counts)
        tmp_counts.columns = ["count"]
        clicks_counts = pd.concat([clicks_counts, tmp_counts]).groupby(level=0).sum()

    count += 1

clicks_counts.index.name = "item_id"
total_clicks = clicks_counts["count"].sum()
clicks_counts["popularity"] = clicks_counts["count"] / total_clicks
clicks_counts["popularity"] = clicks_counts["popularity"].round(5)

print("Calculating item popularity from buys...")
# Buys popularity
buys_counts = buys["item_id"].value_counts().to_frame("count")
buys_counts.index.name = "item_id"
total_buys = buys_counts["count"].sum()
buys_counts["popularity"] = buys_counts["count"] / total_buys
buys_counts["popularity"] = buys_counts["popularity"].round(5)

## Combine All Click Information

print("Combining all click information...")
# Check for duplicate indices
for df, name in zip(
    [result_timing, result_items, result_sequences, result_categories],
    ["result_timing", "result_items", "result_sequences", "result_categories"],
):
    if df.index.duplicated().any():
        print(f"Found duplicate indices in {name}, fixing...")
        # Take the first occurrence of each duplicated index
        df = df[~df.index.duplicated(keep="first")]

# Make sure all dataframes have unique indices before concatenating
result_timing = result_timing[~result_timing.index.duplicated(keep="first")]
result_items = result_items[~result_items.index.duplicated(keep="first")]
result_sequences = result_sequences[~result_sequences.index.duplicated(keep="first")]
result_categories = result_categories[~result_categories.index.duplicated(keep="first")]

# Merge all click analysis DataFrames
clicks_combined = pd.concat(
    [result_timing, result_items, result_sequences, result_categories],
    axis=1,
    join="outer",
)

print("Clicks combined data shape:", clicks_combined.shape)

## Merge with Buy Information and Add Popularity Metrics

print("Merging click and buy information...")
# Merge with buy information (left join to keep all sessions)
training_data = pd.merge(
    clicks_combined,
    buys_transformed["is_buy"],
    how="left",
    left_index=True,
    right_index=True,
)

# Fill missing values (non-buying sessions)
training_data["is_buy"] = training_data["is_buy"].fillna(0)

# Add item popularity for first and last clicked items
training_data = pd.merge(
    training_data,
    clicks_counts["popularity"],
    left_on="first_clicked_item",
    right_index=True,
    how="left",
)
training_data.rename(
    columns={"popularity": "first_clicked_item_popularity"}, inplace=True
)

training_data = pd.merge(
    training_data,
    clicks_counts["popularity"],
    left_on="last_clicked_item",
    right_index=True,
    how="left",
)
training_data.rename(
    columns={"popularity": "last_clicked_item_popularity"}, inplace=True
)

# Fill missing popularities with 0
training_data["first_clicked_item_popularity"] = training_data[
    "first_clicked_item_popularity"
].fillna(0)
training_data["last_clicked_item_popularity"] = training_data[
    "last_clicked_item_popularity"
].fillna(0)

print("Training data preparation complete")
print(f"Training data shape: {training_data.shape}")

## Exploratory Data Analysis

print("Performing exploratory analysis...")

# Check probability of first and last clicked items being purchased
buy_sessions = training_data[training_data["is_buy"] == 1]


# Function to check if item is in purchased items
def check_if_purchased(row):
    if not isinstance(row["unique_items_bought"], list):
        return 0
    if row["item_id"] in row["unique_items_bought"]:
        return 1
    return 0


# Merge buy sessions with unique purchased items
buy_analysis = pd.merge(
    buy_sessions[["first_clicked_item", "last_clicked_item"]],
    buys_transformed["unique_items_bought"],
    left_index=True,
    right_index=True,
    how="inner",
)

# Check first clicked item purchase rate
buy_analysis["first_item_purchased"] = buy_analysis.apply(
    lambda row: 1 if row["first_clicked_item"] in row["unique_items_bought"] else 0,
    axis=1,
)

# Check last clicked item purchase rate
buy_analysis["last_item_purchased"] = buy_analysis.apply(
    lambda row: 1 if row["last_clicked_item"] in row["unique_items_bought"] else 0,
    axis=1,
)

# Plot results
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.barplot(
    x=buy_analysis["first_item_purchased"].value_counts().index,
    y=buy_analysis["first_item_purchased"].value_counts() / len(buy_analysis),
)
plt.title("Probability of First Clicked Item Being Purchased")
plt.xlabel("First Clicked Item Purchased")
plt.ylabel("Probability")

plt.subplot(1, 2, 2)
sns.barplot(
    x=buy_analysis["last_item_purchased"].value_counts().index,
    y=buy_analysis["last_item_purchased"].value_counts() / len(buy_analysis),
)
plt.title("Probability of Last Clicked Item Being Purchased")
plt.xlabel("Last Clicked Item Purchased")
plt.ylabel("Probability")
plt.tight_layout()
plt.show()

# Average dwell time comparison
avg_dwell_buy = training_data[training_data["is_buy"] == 1]["dwell_time_seconds"].mean()
avg_dwell_nobuy = training_data[training_data["is_buy"] == 0][
    "dwell_time_seconds"
].mean()

print(f"Average dwell time for buying sessions: {avg_dwell_buy:.2f} seconds")
print(f"Average dwell time for non-buying sessions: {avg_dwell_nobuy:.2f} seconds")

# Most popular days analysis
plt.figure(figsize=(12, 6))
day_counts = training_data["dayofweek"].value_counts().sort_index()
sns.barplot(x=day_counts.index, y=day_counts.values)
plt.xlabel("Day of Week (0=Monday, 6=Sunday)")
plt.ylabel("Count of Sessions")
plt.title("Most Popular Days Based on Number of Sessions")
plt.xticks(
    range(7),
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
)
plt.show()

# Popular days for buying events
plt.figure(figsize=(12, 6))
buy_day_counts = (
    training_data[training_data["is_buy"] == 1]["dayofweek"].value_counts().sort_index()
)
sns.barplot(x=buy_day_counts.index, y=buy_day_counts.values)
plt.xlabel("Day of Week (0=Monday, 6=Sunday)")
plt.ylabel("Count of Buying Sessions")
plt.title("Popular Days for Buying Events")
plt.xticks(
    range(7),
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
)
plt.show()

# Best time of day for buying events
plt.figure(figsize=(12, 6))
buy_time_counts = training_data[training_data["is_buy"] == 1][
    "timeofday"
].value_counts()
sns.barplot(x=buy_time_counts.index, y=buy_time_counts.values)
plt.xlabel("Time of Day")
plt.ylabel("Count of Buying Sessions")
plt.title("Best Time of Day for Buying Events")
plt.show()

# Class imbalance check
print("Class distribution (Buy vs. Not Buy):")
print(training_data["is_buy"].value_counts())
print(f"Buy ratio: {training_data['is_buy'].mean():.4f}")

## Handle Class Imbalance

print("Handling class imbalance...")


# Function to undersample the majority class
def undersample(df, target_col):
    # Get counts of each class
    class_counts = df[target_col].value_counts()

    # Identify minority class
    minority_class = class_counts.idxmin()
    minority_count = class_counts.min()

    # Get all rows from minority class
    minority_df = df[df[target_col] == minority_class]

    # Sample same number of rows from majority class
    majority_classes = [c for c in class_counts.index if c != minority_class]
    sampled_dfs = [minority_df]

    for cls in majority_classes:
        majority_df = df[df[target_col] == cls]
        sampled_df = majority_df.sample(minority_count, random_state=42)
        sampled_dfs.append(sampled_df)

    # Combine and shuffle
    balanced_df = pd.concat(sampled_dfs).sample(frac=1, random_state=42)

    return balanced_df


# Create a balanced dataset
balanced_training_data = undersample(training_data, "is_buy")

print("Class distribution after balancing:")
print(balanced_training_data["is_buy"].value_counts())

## Feature Engineering

print("Performing feature engineering...")
# One-hot encode time of day
balanced_training_data = pd.get_dummies(
    balanced_training_data, columns=["timeofday"], prefix="", prefix_sep=""
)

# Select features for modeling
feature_cols = [
    "dwell_time_seconds",
    "total_clicks",
    "dayofweek",
    "dayofmonth",
    "hourofclick",
    "click_rate",
    "total_unique_items",
    "total_unique_categories",
    "Number_clicked_visited_categories",
    "Special_offer_click",
    "first_clicked_item_popularity",
    "last_clicked_item_popularity",
    "Late Night",
    "Early Morning",
    "Morning",
    "Noon",
    "Evening",
    "Night",
]

# Some session IDs may not have all features, so we need to handle NaNs
modeling_data = balanced_training_data[feature_cols + ["is_buy"]].copy()
modeling_data = modeling_data.fillna(0)

# Correlation analysis
plt.figure(figsize=(16, 14))
correlation_matrix = modeling_data.corr()
mask = np.triu(correlation_matrix)
sns.heatmap(
    correlation_matrix,
    annot=True,
    mask=mask,
    cmap="coolwarm",
    linewidths=0.5,
    fmt=".2f",
)
plt.title("Feature Correlation Matrix")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

## Prepare Data for Modeling

print("Preparing data for modeling...")
X = modeling_data.drop("is_buy", axis=1)
y = modeling_data["is_buy"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

## Model 1: Logistic Regression

print("Training Logistic Regression model...")
# Create and train the model
log_reg = LogisticRegression(solver="saga", max_iter=1000, random_state=42, n_jobs=-1)
log_reg.fit(X_train, y_train)

# Make predictions
y_pred_proba_lr = log_reg.predict_proba(X_test)[:, 1]
threshold = 0.5
y_pred_lr = (y_pred_proba_lr >= threshold).astype(int)

# Evaluate the model
print("\nLogistic Regression Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
print(f"AUC: {roc_auc_score(y_test, y_pred_proba_lr):.4f}")
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred_lr)
print(
    pd.DataFrame(
        cm,
        columns=["Predicted Not Buy", "Predicted Buy"],
        index=["Actual Not Buy", "Actual Buy"],
    )
)

print("\nClassification Report:")
print(classification_report(y_test, y_pred_lr))

# Plot ROC curve
plt.figure(figsize=(10, 8))
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_lr)
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_pred_proba_lr):.4f}")
plt.plot([0, 1], [0, 1], "k--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Logistic Regression")
plt.legend()
plt.show()

# Find optimal threshold
roc_df = pd.DataFrame(
    {"fpr": fpr, "tpr": tpr, "threshold": thresholds, "difference": tpr - fpr}
)
optimal_threshold = roc_df.loc[roc_df["difference"].idxmax(), "threshold"]
print(f"Optimal threshold: {optimal_threshold:.4f}")


# TODO:

## Model 2: LightGBM

print("\nTraining LightGBM model...")
# Create and train the LightGBM model
gbm = lgb.LGBMClassifier(
    boosting_type="gbdt",
    learning_rate=0.1,
    n_estimators=200,
    num_leaves=32,
    random_state=42,
    n_jobs=-1,
)

# In newer versions of LightGBM, early_stopping_rounds is passed via callbacks
gbm.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    eval_metric=["auc", "binary_logloss"],
    callbacks=[
        lgb.early_stopping(20)
    ],  # Using callbacks instead of early_stopping_rounds
    verbose=50,
)

# Make predictions
y_pred_proba_gbm = gbm.predict_proba(X_test)[:, 1]
y_pred_gbm = (y_pred_proba_gbm >= 0.5).astype(int)

# Evaluate the model
print("\nLightGBM Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_gbm):.4f}")
print(f"AUC: {roc_auc_score(y_test, y_pred_proba_gbm):.4f}")
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred_gbm)
print(
    pd.DataFrame(
        cm,
        columns=["Predicted Not Buy", "Predicted Buy"],
        index=["Actual Not Buy", "Actual Buy"],
    )
)

print("\nClassification Report:")
print(classification_report(y_test, y_pred_gbm))

# Plot feature importance
plt.figure(figsize=(12, 8))
lgb.plot_importance(gbm, max_num_features=15, figsize=(12, 8))
plt.title("Feature Importance - LightGBM")
plt.show()


## Model 3: Random Forest

print("\nTraining Random Forest model...")
# Create and train the Random Forest model
rf = RandomForestClassifier(
    n_estimators=100, max_depth=10, min_samples_split=10, random_state=42, n_jobs=-1
)

rf.fit(X_train, y_train)

# Make predictions
y_pred_proba_rf = rf.predict_proba(X_test)[:, 1]
y_pred_rf = (y_pred_proba_rf >= 0.5).astype(int)

# Evaluate the model
print("\nRandom Forest Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"AUC: {roc_auc_score(y_test, y_pred_proba_rf):.4f}")
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred_rf)
print(
    pd.DataFrame(
        cm,
        columns=["Predicted Not Buy", "Predicted Buy"],
        index=["Actual Not Buy", "Actual Buy"],
    )
)

print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf))

# Plot feature importance
plt.figure(figsize=(12, 8))
feature_importance = pd.DataFrame(
    {"Feature": X.columns, "Importance": rf.feature_importances_}
).sort_values("Importance", ascending=False)

sns.barplot(x="Importance", y="Feature", data=feature_importance.head(15))
plt.title("Feature Importance - Random Forest")
plt.show()

## Model 4: Neural Network (PyTorch)

print("\nTraining Neural Network model with PyTorch...")


# Define PyTorch model
class BinaryClassifier(nn.Module):
    def __init__(self, input_dim):
        super(BinaryClassifier, self).__init__()
        self.layer1 = nn.Linear(input_dim, 32)
        self.dropout1 = nn.Dropout(0.2)
        self.layer2 = nn.Linear(32, 16)
        self.dropout2 = nn.Dropout(0.2)
        self.layer3 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout1(x)
        x = self.relu(self.layer2(x))
        x = self.dropout2(x)
        x = self.sigmoid(self.layer3(x))
        return x


# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert data to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.FloatTensor(y_train.values).reshape(-1, 1)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.FloatTensor(y_test.values).reshape(-1, 1)

# Create dataset and dataloader for training
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# Initialize model, loss and optimizer
model = BinaryClassifier(input_dim=X_train.shape[1])
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 50
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
best_val_loss = float("inf")
best_model_state = None
patience = 10
patience_counter = 0

# Training loop
print("Starting training...")
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    correct_train = 0
    total_train = 0

    for inputs, labels in train_loader:
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * inputs.size(0)
        predicted = (outputs >= 0.5).float()
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    # Calculate training metrics
    train_loss = train_loss / len(train_loader.dataset)
    train_accuracy = correct_train / total_train
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    # Validation
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)
            predicted = (outputs >= 0.5).float()
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    # Calculate validation metrics
    val_loss = val_loss / len(test_loader.dataset)
    val_accuracy = correct_val / total_val
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = model.state_dict().copy()
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break

    if (epoch + 1) % 5 == 0:
        print(
            f"Epoch [{epoch+1}/{epochs}], "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}"
        )

# Load best model
if best_model_state is not None:
    model.load_state_dict(best_model_state)

# Make predictions
model.eval()
with torch.no_grad():
    y_pred_proba_nn = model(X_test_tensor).cpu().numpy().flatten()
    y_pred_nn = (y_pred_proba_nn >= 0.5).astype(int)

# Create history dictionary to match TensorFlow format for later plotting
history = {
    "loss": train_losses,
    "val_loss": val_losses,
    "accuracy": train_accuracies,
    "val_accuracy": val_accuracies,
}

# Evaluate the model
print("\nNeural Network Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_nn):.4f}")
print(f"AUC: {roc_auc_score(y_test, y_pred_proba_nn):.4f}")
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred_nn)
print(
    pd.DataFrame(
        cm,
        columns=["Predicted Not Buy", "Predicted Buy"],
        index=["Actual Not Buy", "Actual Buy"],
    )
)

print("\nClassification Report:")
print(classification_report(y_test, y_pred_nn))

# Plot training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Accuracy Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.show()

## Compare Models

print("\nModel Comparison:")
model_comparison = pd.DataFrame(
    {
        "Model": ["Logistic Regression", "LightGBM", "Random Forest", "Neural Network"],
        "Accuracy": [
            accuracy_score(y_test, y_pred_lr),
            accuracy_score(y_test, y_pred_gbm),
            accuracy_score(y_test, y_pred_rf),
            accuracy_score(y_test, y_pred_nn),
        ],
        "AUC": [
            roc_auc_score(y_test, y_pred_proba_lr),
            roc_auc_score(y_test, y_pred_proba_gbm),
            roc_auc_score(y_test, y_pred_proba_rf),
            roc_auc_score(y_test, y_pred_proba_nn),
        ],
        "Precision": [
            precision_score(y_test, y_pred_lr),
            precision_score(y_test, y_pred_gbm),
            precision_score(y_test, y_pred_rf),
            precision_score(y_test, y_pred_nn),
        ],
        "Recall": [
            recall_score(y_test, y_pred_lr),
            recall_score(y_test, y_pred_gbm),
            recall_score(y_test, y_pred_rf),
            recall_score(y_test, y_pred_nn),
        ],
        "F1 Score": [
            f1_score(y_test, y_pred_lr),
            f1_score(y_test, y_pred_gbm),
            f1_score(y_test, y_pred_rf),
            f1_score(y_test, y_pred_nn),
        ],
    }
)

print(model_comparison.set_index("Model").round(4))

# Plot ROC curves for all models
plt.figure(figsize=(10, 8))

# Logistic Regression
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_proba_lr)
plt.plot(
    fpr_lr,
    tpr_lr,
    label=f"Logistic Regression (AUC = {roc_auc_score(y_test, y_pred_proba_lr):.4f})",
)

# LightGBM
fpr_gbm, tpr_gbm, _ = roc_curve(y_test, y_pred_proba_gbm)
plt.plot(
    fpr_gbm,
    tpr_gbm,
    label=f"LightGBM (AUC = {roc_auc_score(y_test, y_pred_proba_gbm):.4f})",
)

# Random Forest
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_proba_rf)
plt.plot(
    fpr_rf,
    tpr_rf,
    label=f"Random Forest (AUC = {roc_auc_score(y_test, y_pred_proba_rf):.4f})",
)

# Neural Network
fpr_nn, tpr_nn, _ = roc_curve(y_test, y_pred_proba_nn)
plt.plot(
    fpr_nn,
    tpr_nn,
    label=f"Neural Network (AUC = {roc_auc_score(y_test, y_pred_proba_nn):.4f})",
)

# Random baseline
plt.plot([0, 1], [0, 1], "k--", label="Random")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves for All Models")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

## Item Prediction

print("\n## Predicting Which Items Will Be Purchased")

# For this part, we'll use a different approach focusing on session-level item recommendations

# First, prepare a dataset linking sessions, items, and purchase information
print("Preparing item prediction dataset...")

# Create a session item dataset with relevant features
session_item_data = pd.DataFrame()
count = 1

for chunk in pd.read_csv(
    clicks_file,
    names=clicks_columns,
    usecols=["session", "item_id", "category"],
    converters={"category": convert_category},
    chunksize=500000,
):
    print(f"Processing chunk {count} for item prediction...")

    # Group by session and item_id to get counts for each item in each session
    chunk_grouped = (
        chunk.groupby(["session", "item_id"]).size().reset_index(name="click_count")
    )

    # Add category information (using the most common category for each item in the session)
    chunk_categories = (
        chunk.groupby(["session", "item_id"])["category"]
        .apply(lambda x: pd.Series.mode(x)[0] if not pd.Series.mode(x).empty else -1)
        .reset_index()
    )

    # Merge counts and categories
    chunk_data = pd.merge(
        chunk_grouped, chunk_categories, on=["session", "item_id"], how="inner"
    )

    # Collect results
    if session_item_data.empty:
        session_item_data = chunk_data
    else:
        session_item_data = pd.concat([session_item_data, chunk_data])

    count += 1

# Add item popularity from clicks
session_item_data = pd.merge(
    session_item_data,
    clicks_counts["popularity"],
    left_on="item_id",
    right_index=True,
    how="left",
)
session_item_data.rename(columns={"popularity": "click_popularity"}, inplace=True)

# Add item popularity from buys (if available)
session_item_data = pd.merge(
    session_item_data,
    buys_counts["popularity"],
    left_on="item_id",
    right_index=True,
    how="left",
)
session_item_data.rename(columns={"popularity": "buy_popularity"}, inplace=True)
session_item_data["buy_popularity"] = session_item_data["buy_popularity"].fillna(0)

# Get buying sessions
buy_sessions = buys_transformed.index.unique()

# Create a dataset for validation with only buying sessions
validation_data = session_item_data[
    session_item_data["session"].isin(buy_sessions)
].copy()


# Add a label indicating if an item was purchased in each session
def add_purchase_labels(row):
    try:
        session_purchases = buys_transformed.loc[row["session"], "unique_items_bought"]
        if row["item_id"] in session_purchases:
            return 1
        return 0
    except (KeyError, TypeError):
        return 0


validation_data["was_purchased"] = validation_data.apply(add_purchase_labels, axis=1)

print(f"Item prediction dataset created with {len(validation_data)} rows")
print(f"Purchased items: {validation_data['was_purchased'].sum()}")
print(f"Purchase rate: {validation_data['was_purchased'].mean():.4f}")

# Create item-based features for prediction
print("Creating item prediction features...")

# Add session-level features
validation_data["is_first_item"] = (
    validation_data.groupby("session")["click_count"]
    .transform(lambda x: pd.Series(range(len(x))) == 0)
    .astype(int)
)

validation_data["is_last_item"] = (
    validation_data.groupby("session")["click_count"]
    .transform(lambda x: pd.Series(range(len(x))) == len(x) - 1)
    .astype(int)
)

validation_data["item_click_ratio"] = validation_data.groupby("session")[
    "click_count"
].transform(lambda x: x / x.sum())

validation_data["is_special_offer"] = (validation_data["category"] == -1).astype(int)

# Add recency features based on timestamp data from the original clicks
item_recency = pd.DataFrame()
count = 1

for chunk in pd.read_csv(
    clicks_file,
    names=clicks_columns,
    usecols=["session", "item_id", "timestamp"],
    parse_dates=["timestamp"],
    chunksize=500000,
):
    print(f"Processing chunk {count} for recency features...")

    # Get the last time each item was clicked in each session
    chunk_last_click = (
        chunk.groupby(["session", "item_id"])["timestamp"].max().reset_index()
    )

    # Get the session end time
    chunk_session_end = chunk.groupby("session")["timestamp"].max().reset_index()
    chunk_session_end.rename(columns={"timestamp": "session_end"}, inplace=True)

    # Merge to get time since last click
    chunk_recency = pd.merge(
        chunk_last_click, chunk_session_end, on="session", how="inner"
    )

    # Calculate recency (how close to the end of the session)
    chunk_recency["seconds_before_end"] = (
        chunk_recency["session_end"] - chunk_recency["timestamp"]
    ).dt.total_seconds()

    # Collect results
    if item_recency.empty:
        item_recency = chunk_recency[["session", "item_id", "seconds_before_end"]]
    else:
        item_recency = pd.concat(
            [item_recency, chunk_recency[["session", "item_id", "seconds_before_end"]]]
        )

    count += 1

# Merge recency information
validation_data = pd.merge(
    validation_data, item_recency, on=["session", "item_id"], how="left"
)

# Normalize seconds_before_end to get recency score (closer to 1 means more recent)
validation_data["recency_score"] = validation_data.groupby("session")[
    "seconds_before_end"
].transform(lambda x: 1 - (x / x.max() if x.max() > 0 else 0))

print("Item prediction features created")
print(validation_data.head())

## Train Item Prediction Model

print("\nTraining item prediction model...")
# Prepare features for item prediction
item_features = [
    "click_count",
    "category",
    "click_popularity",
    "buy_popularity",
    "is_first_item",
    "is_last_item",
    "item_click_ratio",
    "is_special_offer",
    "recency_score",
]

X_item = validation_data[item_features]
y_item = validation_data["was_purchased"]

# Train-test split for item prediction
X_item_train, X_item_test, y_item_train, y_item_test = train_test_split(
    X_item, y_item, test_size=0.2, random_state=42, stratify=y_item
)

print(
    f"Training set: {X_item_train.shape[0]} samples, Positive ratio: {y_item_train.mean():.4f}"
)
print(
    f"Test set: {X_item_test.shape[0]} samples, Positive ratio: {y_item_test.mean():.4f}"
)

# Train a model to predict item purchases
item_model = lgb.LGBMClassifier(
    boosting_type="gbdt",
    learning_rate=0.1,
    n_estimators=200,
    num_leaves=32,
    class_weight="balanced",
    random_state=42,
)

item_model.fit(
    X_item_train,
    y_item_train,
    eval_set=[(X_item_test, y_item_test)],
    eval_metric=["auc", "binary_logloss"],
    early_stopping_rounds=20,
    verbose=50,
)

# Predict item purchases
y_item_pred_proba = item_model.predict_proba(X_item_test)[:, 1]
y_item_pred = (y_item_pred_proba >= 0.5).astype(int)

# Evaluate the item prediction model
print("\nItem Prediction Model Results:")
print(f"Accuracy: {accuracy_score(y_item_test, y_item_pred):.4f}")
print(f"AUC: {roc_auc_score(y_item_test, y_item_pred_proba):.4f}")
print("\nConfusion Matrix:")
cm = confusion_matrix(y_item_test, y_item_pred)
print(
    pd.DataFrame(
        cm,
        columns=["Predicted Not Purchased", "Predicted Purchased"],
        index=["Actual Not Purchased", "Actual Purchased"],
    )
)

print("\nClassification Report:")
print(classification_report(y_item_test, y_item_pred))

# Plot feature importance for item prediction
plt.figure(figsize=(12, 8))
item_feature_importance = pd.DataFrame(
    {"Feature": item_features, "Importance": item_model.feature_importances_}
).sort_values("Importance", ascending=False)

sns.barplot(x="Importance", y="Feature", data=item_feature_importance)
plt.title("Feature Importance - Item Prediction Model")
plt.show()

## Generate Top-N Recommendations

print("\nGenerating top item recommendations for each session...")


# Function to get top N item recommendations for a session
def get_top_n_recommendations(session_id, n=2):
    # Get items clicked in the session
    session_items = validation_data[validation_data["session"] == session_id]

    if len(session_items) == 0:
        return []

    # Get purchase probabilities for all items
    session_items["purchase_prob"] = item_model.predict_proba(
        session_items[item_features]
    )[:, 1]

    # Return top N items by purchase probability
    return (
        session_items.sort_values("purchase_prob", ascending=False)
        .head(n)["item_id"]
        .tolist()
    )


# Test the recommendation function on some sample sessions
sample_sessions = buy_sessions[:5]
recommendations = {}

for session in sample_sessions:
    recommendations[session] = get_top_n_recommendations(session, n=2)

# Display recommendations
print("\nSample recommendations:")
for session, rec_items in recommendations.items():
    actual_purchases = buys_transformed.loc[session, "unique_items_bought"]
    print(f"Session {session}:")
    print(f"  - Recommended items: {rec_items}")
    print(f"  - Actual purchases: {actual_purchases}")
    print(f"  - Success: {any(item in actual_purchases for item in rec_items)}\n")

# Evaluate recommendations on a larger test set
print("\nEvaluating recommendations on test set...")
test_sessions = np.random.choice(
    buy_sessions, min(1000, len(buy_sessions)), replace=False
)
correct_recommendations = 0
total_recommendations = 0

for session in test_sessions:
    rec_items = get_top_n_recommendations(session, n=2)
    actual_purchases = buys_transformed.loc[session, "unique_items_bought"]

    if any(item in actual_purchases for item in rec_items):
        correct_recommendations += 1
    total_recommendations += 1

# Calculate hit rate
hit_rate = correct_recommendations / total_recommendations
print(f"Hit rate (at least one correct recommendation): {hit_rate:.4f}")

## Alternative Approach: Collaborative Filtering with Item Similarity

print("\nAlternative approach: Item similarity-based recommendations")


# Calculate item co-occurrence in sessions
def calc_item_similarity(item1, item2):
    """
    Calculate item similarity based on co-occurrence and other metrics
    """
    # Get sessions where both items appear
    sessions_item1 = set(
        validation_data[validation_data["item_id"] == item1]["session"]
    )
    sessions_item2 = set(
        validation_data[validation_data["item_id"] == item2]["session"]
    )

    # Calculate Jaccard similarity
    if len(sessions_item1) == 0 or len(sessions_item2) == 0:
        return 0

    common_sessions = sessions_item1.intersection(sessions_item2)
    jaccard_sim = len(common_sessions) / len(sessions_item1.union(sessions_item2))

    return jaccard_sim


# Function to get recommendations based on item similarity
def get_similar_item_recommendations(session_id, n=2):
    # Get items clicked in the session
    session_items = validation_data[validation_data["session"] == session_id][
        "item_id"
    ].unique()

    if len(session_items) == 0:
        return []

    # Get items with highest click_count and recency
    session_details = validation_data[validation_data["session"] == session_id]

    # Combine recency and click count for a weighted score
    session_details["combined_score"] = (
        session_details["recency_score"] * 0.7
        + session_details["item_click_ratio"] * 0.3
    )

    # Get top items from session by combined score
    top_session_items = session_details.sort_values("combined_score", ascending=False)[
        "item_id"
    ].tolist()

    # If only 1 or 2 items in session, just return them
    if len(top_session_items) <= n:
        return top_session_items

    # Otherwise, take the top N
    return top_session_items[:n]


# Test the similarity-based recommendation on sample sessions
similarity_recommendations = {}

for session in sample_sessions:
    similarity_recommendations[session] = get_similar_item_recommendations(session, n=2)

# Display similarity-based recommendations
print("\nSample similarity-based recommendations:")
for session, rec_items in similarity_recommendations.items():
    actual_purchases = buys_transformed.loc[session, "unique_items_bought"]
    print(f"Session {session}:")
    print(f"  - Recommended items: {rec_items}")
    print(f"  - Actual purchases: {actual_purchases}")
    print(f"  - Success: {any(item in actual_purchases for item in rec_items)}\n")

# Evaluate similarity-based recommendations
print("\nEvaluating similarity-based recommendations...")
sim_correct_recommendations = 0
sim_total_recommendations = 0

for session in test_sessions:
    rec_items = get_similar_item_recommendations(session, n=2)
    actual_purchases = buys_transformed.loc[session, "unique_items_bought"]

    if any(item in actual_purchases for item in rec_items):
        sim_correct_recommendations += 1
    sim_total_recommendations += 1

# Calculate hit rate for similarity-based recommendations
sim_hit_rate = sim_correct_recommendations / sim_total_recommendations
print(f"Similarity-based hit rate: {sim_hit_rate:.4f}")

## Conclusion

print("\n## Conclusion")
print(
    "In this notebook, we've developed a comprehensive analytics pipeline for e-commerce clickstream data:"
)
print(
    "1. Loaded and processed large volumes of session data using memory-efficient chunking"
)
print("2. Performed feature engineering to extract meaningful session characteristics")
print("3. Trained models to predict purchasing sessions with high accuracy")
print("4. Developed item recommendation approaches with good hit rates")
print(
    f"5. Best model for session purchase prediction: {model_comparison.loc[model_comparison['AUC'].idxmax(), 'Model']} (AUC: {model_comparison['AUC'].max():.4f})"
)
print(
    f"6. Best recommendation method: {'ML-based' if hit_rate > sim_hit_rate else 'Similarity-based'} (Hit rate: {max(hit_rate, sim_hit_rate):.4f})"
)
print("\nThese models can be used to:")
print("- Target potential buyers with promotions to increase conversion")
print("- Recommend relevant products to increase average order value")
print("- Optimize website layout based on user browsing patterns")
print("- Implement real-time prediction during customer sessions")
