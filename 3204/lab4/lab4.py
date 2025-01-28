import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from optbinning import OptimalBinning
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import auc, classification_report, roc_curve
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data
X, y = make_classification(
    n_samples=2000, n_features=4, n_redundant=0, n_informative=4, random_state=42
)

# Create DataFrame with meaningful column names
feature_names = [
    "text_length",
    "avg_word_length",
    "exclamation_count",
    "question_mark_count",
]
X_df = pd.DataFrame(X, columns=feature_names)

# Add binary sentiment score (0 or 1)
X_df["sentiment_score"] = np.random.choice([0, 1], size=2000)

# Create final dataset
df = X_df.copy()
df["target"] = y

# EDA Section
print("Starting Exploratory Data Analysis...")

# 1. Distribution plots
plt.figure(figsize=(20, 15))
for i, feature in enumerate(feature_names + ["sentiment_score"], 1):
    plt.subplot(3, 2, i)
    if feature == "sentiment_score":
        sns.countplot(data=df, x=feature, hue="target")
        plt.title(f"Distribution of {feature} by Target Class")
    else:
        sns.kdeplot(data=df, x=feature, hue="target", common_norm=False)
        plt.title(f"Distribution of {feature} by Target Class")
    plt.xlabel(feature)
    plt.ylabel("Count" if feature == "sentiment_score" else "Density")

# Correlation heatmap
plt.subplot(3, 2, 6)
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", center=0)
plt.title("Correlation Heatmap")
plt.tight_layout(pad=3.0)
plt.show()

# 2. Box plots
plt.figure(figsize=(20, 15))
for i, feature in enumerate(feature_names, 1):
    plt.subplot(3, 2, i)
    sns.boxplot(data=df, x="target", y=feature)
    plt.title(f"Boxplot of {feature} by Target Class")
    plt.xlabel("Target Class")
    plt.ylabel(feature)

# Summary statistics
plt.subplot(3, 2, 5)
sns.boxplot(data=df, x="target", y="sentiment_score")
plt.title("Boxplot of Sentiment Score by Target Class")
plt.xlabel("Target Class")
plt.ylabel("Sentiment Score")

plt.subplot(3, 2, 6)
plt.axis("off")
summary_stats = df[feature_names + ["sentiment_score"]].describe()
plt.text(0.1, 0.9, "Summary Statistics:", fontsize=12, fontweight="bold")
y_pos = 0.8
for stat in summary_stats.index:
    plt.text(0.1, y_pos, f"\n{stat}:", fontsize=10, fontweight="bold")
    for col in summary_stats.columns:
        plt.text(
            0.1, y_pos - 0.05, f"{col}: {summary_stats.loc[stat, col]:.2f}", fontsize=10
        )
        y_pos -= 0.05
    y_pos -= 0.05

plt.tight_layout(pad=3.0)
plt.show()

# Print additional statistics
print("\nSkewness:")
print(df[feature_names + ["sentiment_score"]].skew())

print("\nKurtosis:")
print(df[feature_names + ["sentiment_score"]].kurtosis())

print("\nClass Distribution:")
print(df["target"].value_counts(normalize=True))

# Feature importance
mi_scores = mutual_info_classif(df[feature_names + ["sentiment_score"]], df["target"])
mi_df = pd.DataFrame(
    {"Feature": feature_names + ["sentiment_score"], "Mutual Information": mi_scores}
)
mi_df = mi_df.sort_values("Mutual Information", ascending=False)

print("\nFeature Importance (Mutual Information):")
print(mi_df)

# Classification Section
print("\nStarting Classification Analysis...")

# Prepare data for classification
X = df[feature_names + ["sentiment_score"]]
y = df["target"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Function to calculate WOE and create bins
def calculate_woe_bins(data, target, feature, n_bins=10):
    if len(data[feature].unique()) <= 2:  # For binary features
        optb = OptimalBinning(name=feature, dtype="categorical", solver="cp")
    else:  # For continuous features
        optb = OptimalBinning(
            name=feature, dtype="numerical", solver="cp", max_bins=n_bins
        )
    optb.fit(data[feature], target)
    binning_table = optb.binning_table.build()
    return optb, binning_table


# Calculate WOE for each feature
woe_tables = []
binning_objects = []

for feature in feature_names + ["sentiment_score"]:
    optb, binning_table = calculate_woe_bins(X_train, y_train, feature)
    woe_tables.append(binning_table)
    binning_objects.append(optb)

# Transform features using WOE
X_train_woe = X_train.copy()
X_test_woe = X_test.copy()

for i, feature in enumerate(feature_names + ["sentiment_score"]):
    X_train_woe[feature] = binning_objects[i].transform(X_train[feature], metric="woe")
    X_test_woe[feature] = binning_objects[i].transform(X_test[feature], metric="woe")

# Random Forest with RandomizedSearchCV
param_dist = {
    "n_estimators": [100, 200, 300, 400, 500],
    "max_depth": [10, 20, 30, 40, 50, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["auto", "sqrt"],
}

# Train with WOE features
rf = RandomForestClassifier(random_state=42)
rf_random = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=100,
    cv=5,
    random_state=42,
    n_jobs=-1,
)
rf_random.fit(X_train_woe, y_train)

# Train without WOE
best_params = rf_random.best_params_
rf_no_woe = RandomForestClassifier(**best_params, random_state=42)
rf_no_woe.fit(X_train, y_train)


# Plot ROC curves
def plot_roc_curves(model, X_train, X_test, y_train, y_test, title):
    plt.figure(figsize=(10, 6))

    # Training set
    y_train_pred = model.predict_proba(X_train)[:, 1]
    fpr_train, tpr_train, _ = roc_curve(y_train, y_train_pred)
    auc_train = auc(fpr_train, tpr_train)
    plt.plot(fpr_train, tpr_train, label=f"Train (AUC = {auc_train:.3f})")

    # Test set
    y_test_pred = model.predict_proba(X_test)[:, 1]
    fpr_test, tpr_test, _ = roc_curve(y_test, y_test_pred)
    auc_test = auc(fpr_test, tpr_test)
    plt.plot(fpr_test, tpr_test, label=f"Test (AUC = {auc_test:.3f})")

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {title}")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Print classification report
    print(f"\nClassification Report - {title}")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))


# Plot ROC curves and print classification reports
plot_roc_curves(
    rf_random.best_estimator_, X_train_woe, X_test_woe, y_train, y_test, "With WOE"
)
plot_roc_curves(rf_no_woe, X_train, X_test, y_train, y_test, "Without WOE")

# Print WOE tables
print("\nWOE Tables:")
for i, feature in enumerate(feature_names + ["sentiment_score"]):
    print(f"\nWOE Table for {feature}:")
    print(woe_tables[i])

print("\nBest parameters found:", rf_random.best_params_)
print("\nFeature importance ranking:")
for feature, importance in zip(
    feature_names + ["sentiment_score"], rf_random.best_estimator_.feature_importances_
):
    print(f"{feature}: {importance:.4f}")
