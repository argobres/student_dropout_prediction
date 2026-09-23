"""Full capstone reproduction: EDA, CV comparison, final evaluation and audit.
Run with python -m student_dropout.workflow from the repository root.
"""
from pathlib import Path
import os
import json
import platform
import importlib.metadata
import joblib
from IPython.display import display

def main():
    from pathlib import Path
    ROOT = Path(__file__).resolve().parents[2]
    RUN_DIR = ROOT / "reports/generated/workflow"
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    os.chdir(RUN_DIR)

    # Report cell 9
    import pandas as pd

    # Report cell 10
    data = pd.read_csv(
        ROOT / "data/raw/data.csv",
        sep=';'
    )
    data

    # Report cell 11
    data.info()

    # Report cell 13
    # Use the checked-in dictionary for offline reproducibility.
    # The original BeautifulSoup retrieval script is in src/student_dropout/scrape_dictionary.py.
    print("Dictionary:", ROOT / "data/reference/variables_table.csv")


    # Report cell 14
    data_dict = pd.read_csv(ROOT / "data/reference/variables_table.csv")

    with pd.option_context(
        "display.max_rows", None,
        "display.max_columns", None,
        "display.max_colwidth", None,
    ):
        display(data_dict)  # Jupyter notebook

    # Report cell 15
    unique_counts = data.nunique().reset_index()
    unique_counts.columns = ["Feature", "Unique Values"]

    unique_counts

    # Report cell 16
    data.describe()

    # Report cell 17
    num_indices = [6, 12, 19, *range(21, 36)]
    num_cols = data.columns[num_indices].tolist()
    categ_cols = [col for col in data.columns if col not in num_cols]
    categ_cols = [col for col in categ_cols if col != "Target"]

    print('Numerical columns:', num_cols, '\n')
    print('Categorical columns:', categ_cols)

    # Report cell 18
    import numpy as np

    # Report cell 20
    import math
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    # Report cell 22
    sns.countplot(data=data, x="Target")
    plt.title("Count Distribution of Target")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("target_count_plot.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Report cell 25
    ncols = 3
    nrows = math.ceil(len(categ_cols) / ncols)

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(18, 5 * nrows), squeeze=False
    )

    for ax, col in zip(axes.flat, categ_cols):
        sns.countplot(data=data, x=col, ax=ax, color="steelblue")
        ax.set_title(col)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=90)

    # Hide unused subplots.
    for ax in list(axes.flat)[len(categ_cols):]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.tight_layout()
    fig.savefig("categorical_count_plots.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Report cell 27
    outcomes = ["Dropout", "Graduate", "Enrolled"]
    colors = ["#d95f59", "#4c9f70", "#e5b44c"]

    ncols = 3
    nrows = math.ceil(len(categ_cols) / ncols)

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(20, 5 * nrows), squeeze=False
    )

    for ax, col in zip(axes.flat, categ_cols):
        percentages = (
            pd.crosstab(data[col], data["Target"], normalize="index")
            .reindex(columns=outcomes, fill_value=0)
            .mul(100)
        )

        percentages.plot(
            kind="bar",
            stacked=True,
            color=colors,
            width=0.8,
            ax=ax,
            legend=False,
        )

        ax.set_title(col)
        ax.set_xlabel("")
        ax.set_ylabel("Students within category (%)")
        ax.set_ylim(0, 100)
        ax.yaxis.set_major_formatter(PercentFormatter(100))
        ax.tick_params(axis="x", rotation=90)

    for ax in list(axes.flat)[len(categ_cols):]:
        ax.set_visible(False)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)

    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig("outcome_percentages_by_category.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Report cell 29
    import pandas as pd

    outcomes = ["Dropout", "Graduate", "Enrolled"]
    min_count = 30
    summaries = []

    for col in categ_cols:
        if col == "Target":
            continue

        categories = data[col].astype("string").fillna("(Missing)")

        counts = pd.crosstab(categories, data["Target"]).reindex(
            columns=outcomes, fill_value=0
        )
        totals = counts.sum(axis=1)
        percentages = counts.div(totals, axis=0).mul(100)

        summary = percentages.rename(
            columns={outcome: f"{outcome} (%)" for outcome in outcomes}
        ).round(1)

        summary.insert(0, "Count", totals)
        summary.insert(0, "Category", summary.index)
        summary.insert(0, "Feature", col)
        summary["Rare group"] = totals < min_count

        summaries.append(summary.reset_index(drop=True))

    eda_summary = pd.concat(summaries, ignore_index=True)
    eda_summary.to_csv("category_outcome_summary.csv", index=False)

    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        display(eda_summary.sample(15))

    # Report cell 31
    summary = pd.read_csv("category_outcome_summary.csv")
    summary["Category"] = summary["Category"].astype(str)

    comparisons = [
        ("Tuition fees up to date", ["0", "1"],
         ["Not up to date", "Up to date"]),
        ("Debtor", ["1", "0"],
         ["Debtor", "Non-debtor"]),
        ("Scholarship holder", ["1", "0"],
         ["Scholarship holder", "Non-holder"]),
        ("Application mode", ["39", "1"],
         ["Mode 39", "Mode 1"]),
        ("Course", ["9130", "9500"],
         ["Course 9130", "Course 9500"]),
        ("Gender", ["0", "1"],
         ["Category 0", "Category 1"]),
    ]

    outcomes = ["Dropout", "Graduate", "Enrolled"]
    colors = ["#d95f59", "#4c9f70", "#e5b44c"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    width = 0.25

    for ax, (feature, categories, labels) in zip(axes.flat, comparisons):
        subset = (
            summary.loc[summary["Feature"] == feature]
            .set_index("Category")
            .loc[categories]
        )

        x = np.arange(len(categories))

        for i, (outcome, color) in enumerate(zip(outcomes, colors)):
            bars = ax.bar(
                x + (i - 1) * width,
                subset[f"{outcome} (%)"],
                width=width,
                label=outcome,
                color=color,
            )
            ax.bar_label(
                bars,
                labels=[f"{v:.1f}%" for v in subset[f"{outcome} (%)"]],
                padding=3,
                fontsize=9,
            )

        ax.set_xticks(x)
        ax.set_xticklabels([
            f"{label}\n(n={int(count):,})"
            for label, count in zip(labels, subset["Count"])
        ])
        ax.set_title(feature)
        ax.set_ylabel("Students within category (%)")
        ax.set_ylim(0, 105)
        ax.set_yticks(range(0, 101, 20))
        ax.yaxis.set_major_formatter(PercentFormatter(100))
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.2)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(
        "clustered_outcome_comparisons.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    # Report cell 32
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    comparisons = [
        ("Tuition fees up to date", [0, 1],
         ["Not up to date", "Up to date"]),
        ("Debtor", [1, 0],
         ["Debtor", "Non-debtor"]),
        ("Scholarship holder", [1, 0],
         ["Scholarship holder", "Non-holder"]),
        ("Application mode", [39, 1],
         ["Mode 39", "Mode 1"]),
        ("Course", [9130, 9500],
         ["Course 9130", "Course 9500"]),
        ("Gender", [0, 1],
         ["Category 0", "Category 1"]),
    ]

    outcomes = ["Dropout", "Graduate", "Enrolled"]
    colors = ["#d95f59", "#4c9f70", "#e5b44c"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    width = 0.25

    for ax, (feature, categories, labels) in zip(axes.flat, comparisons):
        counts = pd.crosstab(data[feature], data["Target"]).reindex(
            index=categories, columns=outcomes, fill_value=0
        )
        totals = counts.sum(axis=1)
        x = np.arange(len(categories))

        for i, (outcome, color) in enumerate(zip(outcomes, colors)):
            bars = ax.bar(
                x + (i - 1) * width,
                counts[outcome],
                width=width,
                label=outcome,
                color=color,
            )
            ax.bar_label(bars, fmt="%.0f", padding=3, fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels([
            f"{label}\n(n={int(count):,})"
            for label, count in zip(labels, totals)
        ])
        ax.set_title(feature)
        ax.set_ylabel("Number of students")
        ax.margins(y=0.15)
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.2)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(
        "clustered_outcome_counts.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    # Report cell 35
    import math
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")

    output_dir = Path("eda_numerical")
    output_dir.mkdir(exist_ok=True)

    # Zero-based positions for the original dataset column order.
    num_indices = [6, 12, 19, *range(21, 36)]
    num_cols = data.columns[num_indices].tolist()

    # Work on a copy and preserve the original dataset.
    numeric_data = data[num_cols].apply(pd.to_numeric, errors="raise")

    outcomes = ["Dropout", "Graduate", "Enrolled"]
    palette = {
        "Dropout": "#d95f59",
        "Graduate": "#4c9f70",
        "Enrolled": "#e5b44c",
    }

    print(f"Numerical features: {len(num_cols)}")
    display(pd.DataFrame({"Numerical feature": num_cols}))

    # Report cell 37
    numerical_summary = numeric_data.describe().T.rename(
        columns={
            "count": "Non-missing count",
            "mean": "Mean",
            "std": "Standard deviation",
            "min": "Minimum",
            "25%": "Q1",
            "50%": "Median",
            "75%": "Q3",
            "max": "Maximum",
        }
    )

    numerical_summary["Missing count"] = numeric_data.isna().sum()
    numerical_summary["Missing (%)"] = numeric_data.isna().mean() * 100
    numerical_summary["Unique values"] = numeric_data.nunique()
    numerical_summary["IQR"] = (
        numerical_summary["Q3"] - numerical_summary["Q1"]
    )
    numerical_summary["Skewness"] = numeric_data.skew()

    with pd.option_context("display.max_columns", None):
        display(numerical_summary.round(2))

    numerical_summary.to_csv(output_dir / "numerical_summary.csv")

    # Report cell 39
    ncols = 3
    nrows = math.ceil(len(num_cols) / ncols)

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(18, 4 * nrows), squeeze=False
    )

    for ax, col in zip(axes.flat, num_cols):
        values = numeric_data[col].dropna()

        # Use one bin per integer when the integer range is manageable.
        is_integer = (
            not values.empty
            and np.allclose(values, np.round(values))
        )

        if is_integer and values.max() - values.min() <= 60:
            bins = np.arange(values.min() - 0.5, values.max() + 1.5, 1)
        else:
            bins = 30

        sns.histplot(values, bins=bins, color="steelblue", ax=ax)
        ax.set_title(col, wrap=True)
        ax.set_xlabel("")
        ax.set_ylabel("Count")

    for ax in axes.flat[len(num_cols):]:
        ax.set_visible(False)

    fig.tight_layout()
    fig.savefig(
        output_dir / "numerical_histograms.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    # Report cell 41
    q1 = numeric_data.quantile(0.25)
    q3 = numeric_data.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outlier_mask = (
        numeric_data.lt(lower_bound, axis="columns")
        | numeric_data.gt(upper_bound, axis="columns")
    )

    outlier_summary = pd.DataFrame({
        "Lower bound": lower_bound,
        "Upper bound": upper_bound,
        "Potential outlier count": outlier_mask.sum(),
        "Potential outlier (%)": (
            outlier_mask.sum()
            .div(numeric_data.count().replace(0, np.nan))
            .mul(100)
        ),
        "Zero IQR": iqr.eq(0),
    }).sort_values("Potential outlier (%)", ascending=False)

    display(outlier_summary.round(2))
    outlier_summary.to_csv(output_dir / "potential_outliers.csv")

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(18, 3.5 * nrows), squeeze=False
    )

    for ax, col in zip(axes.flat, num_cols):
        sns.boxplot(
            x=numeric_data[col],
            color="lightsteelblue",
            ax=ax,
        )
        ax.set_title(col, wrap=True)
        ax.set_xlabel("")

    for ax in axes.flat[len(num_cols):]:
        ax.set_visible(False)

    fig.tight_layout()
    fig.savefig(
        output_dir / "numerical_boxplots.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    # Report cell 43
    plot_data = numeric_data.copy()
    plot_data["Target"] = data["Target"]

    long_data = plot_data.melt(
        id_vars="Target",
        value_vars=num_cols,
        var_name="Feature",
        value_name="Value",
    )

    outcome_summary = (
        long_data.groupby(["Feature", "Target"], observed=True)["Value"]
        .agg(
            Count="count",
            Mean="mean",
            Median="median",
            Q1=lambda values: values.quantile(0.25),
            Q3=lambda values: values.quantile(0.75),
        )
        .reindex(pd.MultiIndex.from_product(
            [num_cols, outcomes],
            names=["Feature", "Target"],
        ))
    )

    with pd.option_context("display.max_rows", None):
        display(outcome_summary.round(2))

    outcome_summary.to_csv(output_dir / "numerical_summary_by_target.csv")

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(18, 4.5 * nrows), squeeze=False
    )

    for ax, col in zip(axes.flat, num_cols):
        sns.boxplot(
            data=plot_data,
            x="Target",
            y=col,
            order=outcomes,
            hue="Target",
            hue_order=outcomes,
            palette=palette,
            dodge=False,
            ax=ax,
        )

        if ax.get_legend() is not None:
            ax.get_legend().remove()

        ax.set_title(col, wrap=True)
        ax.set_xlabel("")
        ax.set_ylabel("Value")

    for ax in axes.flat[len(num_cols):]:
        ax.set_visible(False)

    fig.tight_layout()
    fig.savefig(
        output_dir / "numerical_features_by_target.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    # Report cell 45
    correlation = numeric_data.corr(method="spearman")

    # Short labels keep the heatmap readable.
    feature_ids = [f"N{i + 1}" for i in range(len(num_cols))]
    feature_key = pd.DataFrame({
        "ID": feature_ids,
        "Feature": num_cols,
    })
    display(feature_key)

    heatmap_data = correlation.copy()
    heatmap_data.index = feature_ids
    heatmap_data.columns = feature_ids

    mask = np.triu(
        np.ones_like(heatmap_data, dtype=bool),
        k=1,
    )

    fig, ax = plt.subplots(figsize=(14, 12))

    sns.heatmap(
        heatmap_data,
        mask=mask,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        center=0,
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": 8},
        square=True,
        cbar_kws={"label": "Spearman correlation"},
        ax=ax,
    )

    ax.set_title("Correlations Between Numerical Features")
    fig.tight_layout()
    fig.savefig(
        output_dir / "numerical_correlations.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    # Unique feature pairs, excluding self-correlations.
    upper_triangle = np.triu(
        np.ones(correlation.shape, dtype=bool),
        k=1,
    )

    correlation_pairs = (
        correlation.where(upper_triangle)
        .stack()
        .rename("Spearman correlation")
        .reset_index()
    )
    correlation_pairs.columns = [
        "Feature 1", "Feature 2", "Spearman correlation"
    ]
    correlation_pairs["Absolute correlation"] = (
        correlation_pairs["Spearman correlation"].abs()
    )
    correlation_pairs = correlation_pairs.sort_values(
        "Absolute correlation",
        ascending=False,
    )

    display(correlation_pairs.head(15).round(3))

    correlation.to_csv(output_dir / "spearman_correlations.csv")
    correlation_pairs.to_csv(
        output_dir / "correlation_pairs.csv", index=False
    )
    feature_key.to_csv(output_dir / "correlation_feature_key.csv", index=False)

    # Report cell 48
    from sklearn.base import clone
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import (
        LabelEncoder, OneHotEncoder, StandardScaler, FunctionTransformer
    )
    from sklearn.impute import SimpleImputer
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import SelectFromModel
    from sklearn.inspection import permutation_importance
    from sklearn.decomposition import PCA
    from sklearn.metrics import classification_report, f1_score

    # Report cell 49
    SEED = 42
    output_dir = Path("preprocessing_outputs")
    output_dir.mkdir(exist_ok=True)

    sns.set_theme(style="whitegrid")

    df = data.copy()
    df.columns = df.columns.str.strip()

    if df.columns.duplicated().any():
        raise ValueError("Duplicate column names found after stripping whitespace.")

    if df["Target"].isna().any():
        raise ValueError("Missing Target values must be investigated first.")

    df["Target"] = df["Target"].astype(str).str.strip()

    expected_targets = {"Dropout", "Graduate", "Enrolled"}
    if set(df["Target"]) != expected_targets:
        raise ValueError("Unexpected Target labels.")

    # Explicit names avoid dependence on column positions.
    num_cols = [
        "Previous qualification (grade)",
        "Admission grade",
        "Age at enrollment",
        *[
            f"Curricular units {semester} sem ({measure})"
            for semester in ["1st", "2nd"]
            for measure in [
                "credited", "enrolled", "evaluations",
                "approved", "grade", "without evaluations"
            ]
        ],
        "Unemployment rate",
        "Inflation rate",
        "GDP",
    ]

    missing_columns = set(num_cols) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing expected columns: {missing_columns}")

    categ_cols = [
        col for col in df.columns
        if col not in num_cols + ["Target"]
    ]

    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="raise")

    if np.isinf(df[num_cols].to_numpy(dtype=float)).any():
        raise ValueError("Infinite numerical values found.")

    print("Original numerical features:", len(num_cols))
    print("Original categorical features:", len(categ_cols))
    print("Exact duplicate rows:", df.duplicated().sum())
    # Do not automatically delete identical rows: they may be different students.

    X = df.drop(columns="Target")
    y = df["Target"]

    X_train, X_remaining, y_train_text, y_remaining = train_test_split(
        X, y,
        test_size=0.40,
        stratify=y,
        random_state=SEED,
    )

    X_valid, X_test, y_valid_text, y_test_text = train_test_split(
        X_remaining, y_remaining,
        test_size=0.50,
        stratify=y_remaining,
        random_state=SEED,
    )

    target_encoder = LabelEncoder()
    y_train = target_encoder.fit_transform(y_train_text)
    y_valid = target_encoder.transform(y_valid_text)
    y_test = target_encoder.transform(y_test_text)

    display(pd.DataFrame({
        "Target": target_encoder.classes_,
        "Encoded label": target_encoder.transform(target_encoder.classes_),
    }))

    split_summary = pd.DataFrame({
        "Training": y_train_text.value_counts(),
        "Validation": y_valid_text.value_counts(),
        "Test": y_test_text.value_counts(),
    }).reindex(["Dropout", "Graduate", "Enrolled"])

    display(split_summary)

    # Report cell 51
    ratio_cols = [
        "Approval ratio 1st sem",
        "Approval ratio 2nd sem",
        "Without evaluation ratio 1st sem",
        "Without evaluation ratio 2nd sem",
    ]

    derived_num_cols = ratio_cols + [
        "Grade change",
        "Approved units change",
    ]

    derived_cat_cols = [
        "Age group",
        "Zero approved 1st sem",
        "Zero approved 2nd sem",
        "Zero enrolled 1st sem",
        "Zero enrolled 2nd sem",
    ]

    all_num_cols = num_cols + derived_num_cols
    all_cat_cols = categ_cols + derived_cat_cols


    from student_dropout.features import engineer_features


    X_train_engineered = engineer_features(X_train)

    display(X_train_engineered[derived_num_cols + derived_cat_cols].head())

    # Investigate ratios above one; do not silently clip them.
    ratio_checks = pd.DataFrame({
        "Undefined count": X_train_engineered[ratio_cols].isna().sum(),
        "Below zero count": X_train_engineered[ratio_cols].lt(0).sum(),
        "Above one count": X_train_engineered[ratio_cols].gt(1).sum(),
    })

    display(ratio_checks)

    # Report cell 53
    outcomes = ["Dropout", "Graduate", "Enrolled"]
    palette = {
        "Dropout": "#d95f59",
        "Graduate": "#4c9f70",
        "Enrolled": "#e5b44c",
    }

    eda_train = X_train_engineered.copy()
    eda_train["Target"] = y_train_text.to_numpy()

    # Distributions of the six engineered numerical features.
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))

    for ax, col in zip(axes.flat, derived_num_cols):
        sns.histplot(
            data=eda_train, x=col,
            bins=25, color="steelblue", ax=ax
        )
        ax.set_title(col)
        ax.set_xlabel("")

    fig.tight_layout()
    fig.savefig(
        output_dir / "engineered_distributions.png",
        dpi=300, bbox_inches="tight"
    )
    plt.show()

    # Relationships with Target.
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for ax, col in zip(axes.flat, derived_num_cols):
        sns.boxplot(
            data=eda_train,
            x="Target", y=col,
            order=outcomes,
            hue="Target", hue_order=outcomes,
            palette=palette,
            dodge=False,
            ax=ax,
        )
        if ax.get_legend() is not None:
            ax.get_legend().remove()

        ax.set_title(col)
        ax.set_xlabel("")
        ax.set_ylabel("Value")

    fig.tight_layout()
    fig.savefig(
        output_dir / "engineered_features_by_target.png",
        dpi=300, bbox_inches="tight"
    )
    plt.show()

    age_counts = pd.crosstab(
        eda_train["Age group"], eda_train["Target"]
    ).reindex(
        index=["Under 20", "20–24", "25–34", "35+"],
        columns=outcomes,
        fill_value=0,
    )

    age_summary = age_counts.div(
        age_counts.sum(axis=1).replace(0, np.nan), axis=0
    ).mul(100).round(1)

    age_summary.columns = [f"{col} (%)" for col in age_summary.columns]
    age_summary.insert(0, "Count", age_counts.sum(axis=1))

    display(age_summary)
    age_summary.to_csv(output_dir / "training_age_group_summary.csv")

    # Report cell 55
    numeric_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                add_indicator=False,  # Explicit zero-enrollment indicators remain.
                keep_empty_features=True,
            ),
        ),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="constant",
                fill_value="Missing",
                keep_empty_features=True,
            ),
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="infrequent_if_exist",
                min_frequency=10,
                sparse_output=False,
            ),
        ),
    ])

    # Retain all original and engineered features.
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, all_num_cols),
            ("cat", categorical_pipeline, all_cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    # Unfitted template for later cross-validation.
    preparation_template = Pipeline([
        (
            "engineer",
            FunctionTransformer(engineer_features, validate=False),
        ),
        ("preprocess", preprocessor),
    ])

    # Fit only on training data.
    preparation = clone(preparation_template)

    X_train_prepared = preparation.fit_transform(X_train)
    X_valid_prepared = preparation.transform(X_valid)
    X_test_prepared = preparation.transform(X_test)

    feature_names = preparation.named_steps[
        "preprocess"
    ].get_feature_names_out()

    assert np.isfinite(X_train_prepared).all()
    assert np.isfinite(X_valid_prepared).all()
    assert np.isfinite(X_test_prepared).all()

    print("Training:", X_train_prepared.shape)
    print("Validation:", X_valid_prepared.shape)
    print("Test:", X_test_prepared.shape)

    display(pd.DataFrame(
        X_train_prepared[:5],
        columns=feature_names,
    ))

    # Report cell 57
    exploratory_forest = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )

    exploratory_model = Pipeline([
        ("prepare", clone(preparation_template)),
        ("forest", exploratory_forest),
    ])

    exploratory_model.fit(X_train, y_train)

    valid_predictions = exploratory_model.predict(X_valid)

    print(
        "Exploratory validation macro F1:",
        round(f1_score(y_valid, valid_predictions, average="macro"), 3)
    )

    print(classification_report(
        y_valid,
        valid_predictions,
        labels=np.arange(len(target_encoder.classes_)),
        target_names=target_encoder.classes_,
        zero_division=0,
    ))

    # Permute original columns, before engineering and encoding.
    permutation_result = permutation_importance(
        exploratory_model,
        X_valid,
        y_valid,
        scoring="f1_macro",
        n_repeats=5,
        random_state=SEED,
        n_jobs=1,  # Avoid nested parallelism with the random forest.
    )

    permutation_table = pd.DataFrame({
        "Feature": X_valid.columns,
        "Importance": permutation_result.importances_mean,
        "Repeat SD": permutation_result.importances_std,
    }).sort_values("Importance", ascending=False)

    display(permutation_table)

    top_importance = permutation_table.head(20).sort_values("Importance")

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(
        top_importance["Feature"],
        top_importance["Importance"],
        xerr=top_importance["Repeat SD"],
        color="steelblue",
        capsize=3,
    )
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Decrease in validation macro F1")
    ax.set_title("Original-Feature Permutation Importance")
    fig.tight_layout()
    fig.savefig(
        output_dir / "permutation_importance.png",
        dpi=300, bbox_inches="tight"
    )
    plt.show()

    permutation_table.to_csv(
        output_dir / "permutation_importance.csv", index=False
    )

    # Report cell 59
    selector_template = SelectFromModel(
        estimator=RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        ),
        threshold="mean",
    )

    selector = clone(selector_template)
    selector.fit(X_train_prepared, y_train)

    X_train_selected = selector.transform(X_train_prepared)
    X_valid_selected = selector.transform(X_valid_prepared)
    X_test_selected = selector.transform(X_test_prepared)

    selected_mask = selector.get_support()
    selected_feature_names = feature_names[selected_mask]

    selection_table = pd.DataFrame({
        "Feature": feature_names,
        "Training impurity importance": selector.estimator_.feature_importances_,
        "Selected": selected_mask,
    }).sort_values("Training impurity importance", ascending=False)

    print("Features before selection:", len(feature_names))
    print("Features after selection:", len(selected_feature_names))
    display(selection_table.loc[selection_table["Selected"]])

    selection_table.to_csv(
        output_dir / "embedded_feature_selection.csv", index=False
    )

    # Report cell 61
    pca = PCA(n_components=0.95, svd_solver="full")

    X_train_pca = pca.fit_transform(X_train_prepared)
    X_valid_pca = pca.transform(X_valid_prepared)
    X_test_pca = pca.transform(X_test_prepared)

    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

    print("Original transformed features:", X_train_prepared.shape[1])
    print("PCA components retained:", pca.n_components_)
    print(f"Variance retained: {cumulative_variance[-1]:.2%}")

    pca_variance = pd.DataFrame({
        "Component": np.arange(1, pca.n_components_ + 1),
        "Explained variance": pca.explained_variance_ratio_,
        "Cumulative variance": cumulative_variance,
    })

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(
        pca_variance["Component"],
        cumulative_variance,
        marker="o",
        markersize=3,
    )
    ax.axhline(0.95, color="red", linestyle="--", label="95% variance")
    ax.set_xlabel("Number of principal components")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_title("PCA Variance Retention")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        output_dir / "pca_variance.png",
        dpi=300, bbox_inches="tight"
    )
    plt.show()

    if X_train_pca.shape[1] >= 2:
        pca_plot = pd.DataFrame({
            "PC1": X_train_pca[:, 0],
            "PC2": X_train_pca[:, 1],
            "Target": y_train_text.to_numpy(),
        })

        fig, ax = plt.subplots(figsize=(9, 7))
        sns.scatterplot(
            data=pca_plot,
            x="PC1", y="PC2",
            hue="Target",
            hue_order=outcomes,
            palette=palette,
            alpha=0.45,
            s=25,
            ax=ax,
        )
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        ax.set_title("Training Data in the First Two Principal Components")
        fig.tight_layout()
        fig.savefig(
            output_dir / "pca_target_visualization.png",
            dpi=300, bbox_inches="tight"
        )
        plt.show()

    # Inspect the strongest component coefficients.
    pca_coefficients = pd.DataFrame(
        pca.components_.T,
        index=feature_names,
        columns=[f"PC{i + 1}" for i in range(pca.n_components_)],
    )

    for component in pca_coefficients.columns[:2]:
        strongest = (
            pca_coefficients[component]
            .abs()
            .sort_values(ascending=False)
            .head(10)
            .index
        )
        print(f"\nLargest absolute coefficients for {component}")
        display(pca_coefficients.loc[strongest, [component]])

    pca_variance.to_csv(output_dir / "pca_variance.csv", index=False)
    pca_coefficients.to_csv(output_dir / "pca_coefficients.csv")

    # Report cell 63
    representation_summary = pd.DataFrame({
        "Representation": ["Full", "Selected", "PCA"],
        "Training rows": [
            X_train_prepared.shape[0],
            X_train_selected.shape[0],
            X_train_pca.shape[0],
        ],
        "Features/components": [
            X_train_prepared.shape[1],
            X_train_selected.shape[1],
            X_train_pca.shape[1],
        ],
    })

    display(representation_summary)

    # Full original and engineered features.
    full_feature_pipeline = clone(preparation_template)

    # Embedded feature selection, fitted within each training fold.
    selected_feature_pipeline = Pipeline([
        ("prepare", clone(preparation_template)),
        ("select", clone(selector_template)),
    ])

    # PCA, fitted within each training fold.
    pca_feature_pipeline = Pipeline([
        ("prepare", clone(preparation_template)),
        ("pca", PCA(n_components=0.95, svd_solver="full")),
    ])

    # Report cell 66
    import time
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from pathlib import Path
    from sklearn.base import clone
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import (
        StratifiedKFold,
        GridSearchCV,
        ParameterGrid,
    )
    from sklearn.dummy import DummyClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import (
        RandomForestClassifier,
        GradientBoostingClassifier,
    )
    from sklearn.svm import SVC
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        classification_report,
        ConfusionMatrixDisplay,
    )

    SEED = 42
    N_JOBS = 2  # Reduce to 1 if memory or CPU use is excessive.

    model_output_dir = Path("model_outputs")
    model_output_dir.mkdir(exist_ok=True)

    class_names = target_encoder.classes_
    class_labels = np.arange(len(class_names))

    # Materialize the splits so every experiment uses identical folds.
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED,
    )
    cv_splits = list(cv.split(X_train, y_train))

    scoring = {
        "macro_f1": "f1_macro",
        "balanced_accuracy": "balanced_accuracy",
        "accuracy": "accuracy",
    }

    # Clone the unfitted feature pipelines from the preceding cell.
    representations = {
        "Full": clone(full_feature_pipeline),
        "Selected": clone(selected_feature_pipeline),
        "PCA": clone(pca_feature_pipeline),
    }

    # Avoid nested parallelism: GridSearchCV handles parallel execution.
    representations["Selected"].set_params(
        select__estimator__n_jobs=1
    )

    print("Training rows:", len(X_train))
    print("Validation rows:", len(X_valid))
    print("Test rows reserved:", len(X_test))
    print("Classes:", list(class_names))

    # Report cell 68
    model_specs = {
        "Dummy": {
            "estimator": DummyClassifier(strategy="most_frequent"),
            "representations": ["Full"],
            "grid": {},
        },

        "Logistic regression": {
            "estimator": LogisticRegression(
                solver="lbfgs",
                max_iter=5000,
                random_state=SEED,
            ),
            "representations": ["Full", "Selected", "PCA"],
            "grid": {
                "model__C": [0.1, 1.0, 10.0],
                "model__class_weight": [None, "balanced"],
            },
        },

        "Random forest": {
            "estimator": RandomForestClassifier(
                n_estimators=200,
                max_features="sqrt",
                random_state=SEED,
                n_jobs=1,
            ),
            "representations": ["Full", "Selected"],
            "grid": {
                "model__max_depth": [None, 12],
                "model__min_samples_leaf": [2, 5],
                "model__class_weight": [None, "balanced"],
            },
        },

        "Gradient boosting": {
            "estimator": GradientBoostingClassifier(
                n_estimators=150,
                min_samples_leaf=5,
                random_state=SEED,
            ),
            "representations": ["Full", "Selected"],
            "grid": {
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [2, 3],
            },
        },

        "RBF SVM": {
            "estimator": SVC(
                kernel="rbf",
                probability=False,
                cache_size=512,
            ),
            "representations": ["Full", "PCA"],
            "grid": {
                "model__C": [1.0, 10.0],
                "model__gamma": ["scale", 0.01],
                "model__class_weight": [None, "balanced"],
            },
        },
    }

    total_cv_fits = sum(
        len(spec["representations"])
        * len(ParameterGrid(spec["grid"]))
        * len(cv_splits)
        for spec in model_specs.values()
    )

    print(f"Planned cross-validation fits: {total_cv_fits}")
    print("Additional refits train each experiment's best configuration.")

    # Report cell 70
    searches = {}
    validation_predictions = {}
    comparison_rows = []


    def evaluate_predictions(y_true, y_pred):
        report = classification_report(
            y_true,
            y_pred,
            labels=class_labels,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )

        return {
            "Macro F1": f1_score(
                y_true, y_pred,
                labels=class_labels,
                average="macro",
                zero_division=0,
            ),
            "Accuracy": accuracy_score(y_true, y_pred),
            "Balanced accuracy": balanced_accuracy_score(y_true, y_pred),
            "Dropout precision": report["Dropout"]["precision"],
            "Dropout recall": report["Dropout"]["recall"],
            "Enrolled F1": report["Enrolled"]["f1-score"],
            "Enrolled recall": report["Enrolled"]["recall"],
        }


    for model_name, spec in model_specs.items():
        for representation_name in spec["representations"]:
            experiment = f"{model_name} | {representation_name}"
            print(f"\nRunning: {experiment}", flush=True)
            start = time.perf_counter()

            pipeline = Pipeline([
                ("features", clone(representations[representation_name])),
                ("model", clone(spec["estimator"])),
            ])

            search = GridSearchCV(
                estimator=pipeline,
                param_grid=spec["grid"],
                scoring=scoring,
                refit="macro_f1",
                cv=cv_splits,
                n_jobs=N_JOBS,
                pre_dispatch=N_JOBS,
                return_train_score=True,
                error_score="raise",
            )

            # Raw predictors enter the pipeline.
            search.fit(X_train, y_train)

            elapsed = time.perf_counter() - start
            predictions = search.predict(X_valid)
            validation_metrics = evaluate_predictions(y_valid, predictions)

            searches[experiment] = search
            validation_predictions[experiment] = predictions

            best_index = search.best_index_
            results = search.cv_results_

            comparison_rows.append({
                "Experiment": experiment,
                "Model": model_name,
                "Representation": representation_name,
                "CV macro F1": results["mean_test_macro_f1"][best_index],
                "CV macro F1 SD": results["std_test_macro_f1"][best_index],
                "Training macro F1": results["mean_train_macro_f1"][best_index],
                "CV balanced accuracy": (
                    results["mean_test_balanced_accuracy"][best_index]
                ),
                "Validation macro F1": validation_metrics["Macro F1"],
                "Validation accuracy": validation_metrics["Accuracy"],
                "Validation balanced accuracy": (
                    validation_metrics["Balanced accuracy"]
                ),
                "Validation Dropout precision": (
                    validation_metrics["Dropout precision"]
                ),
                "Validation Dropout recall": (
                    validation_metrics["Dropout recall"]
                ),
                "Validation Enrolled F1": validation_metrics["Enrolled F1"],
                "Validation Enrolled recall": (
                    validation_metrics["Enrolled recall"]
                ),
                "Search seconds": elapsed,
                "Best parameters": search.best_params_,
            })

            safe_name = (
                experiment.lower()
                .replace(" | ", "_")
                .replace(" ", "_")
            )

            pd.DataFrame(results).to_csv(
                model_output_dir / f"cv_results_{safe_name}.csv",
                index=False,
            )

            print(
                f"CV macro F1: {search.best_score_:.3f} | "
                f"Validation macro F1: {validation_metrics['Macro F1']:.3f} | "
                f"Time: {elapsed:.1f}s",
                flush=True,
            )

    comparison = (
        pd.DataFrame(comparison_rows)
        .sort_values("CV macro F1", ascending=False, kind="stable")
        .reset_index(drop=True)
    )

    comparison["Training–CV gap"] = (
        comparison["Training macro F1"] - comparison["CV macro F1"]
    )

    comparison.to_csv(
        model_output_dir / "model_comparison.csv",
        index=False,
    )

    display_columns = [
        "Experiment",
        "CV macro F1",
        "CV macro F1 SD",
        "Validation macro F1",
        "Validation Dropout recall",
        "Validation Enrolled F1",
        "Training–CV gap",
        "Search seconds",
    ]

    display(comparison[display_columns].round(3))

    # Report cell 71
    plot_table = comparison.sort_values("CV macro F1")
    positions = np.arange(len(plot_table))
    height = 0.36

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.barh(
        positions - height / 2,
        plot_table["CV macro F1"],
        height=height,
        xerr=plot_table["CV macro F1 SD"],
        capsize=3,
        label="Training CV mean ± fold SD",
        color="steelblue",
    )

    ax.barh(
        positions + height / 2,
        plot_table["Validation macro F1"],
        height=height,
        label="Validation",
        color="darkorange",
    )

    ax.set_yticks(positions)
    ax.set_yticklabels(plot_table["Experiment"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Macro F1")
    ax.set_title("Model and Feature-Representation Comparison")
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(
        model_output_dir / "model_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    # Report cell 72
    representation_comparison = comparison.pivot(
        index="Model",
        columns="Representation",
        values="CV macro F1",
    )

    display(representation_comparison.round(3))

    # Report cell 74
    # comparison is already sorted by training CV macro F1.
    best_per_algorithm = comparison.drop_duplicates("Model")

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))

    for ax, (_, row) in zip(axes.flat, best_per_algorithm.iterrows()):
        experiment = row["Experiment"]

        ConfusionMatrixDisplay.from_predictions(
            y_valid,
            validation_predictions[experiment],
            labels=class_labels,
            display_labels=class_names,
            normalize="true",
            values_format=".2f",
            cmap="Blues",
            colorbar=False,
            ax=ax,
        )

        ax.set_title(experiment)

    for ax in axes.flat[len(best_per_algorithm):]:
        ax.set_visible(False)

    fig.tight_layout()
    fig.savefig(
        model_output_dir / "validation_confusion_matrices.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    # Report cell 76
    winner_name = comparison.iloc[0]["Experiment"]
    winner_search = searches[winner_name]

    dummy_cv = comparison.loc[
        comparison["Model"].eq("Dummy"), "CV macro F1"
    ].iloc[0]

    if winner_search.best_score_ <= dummy_cv:
        raise RuntimeError(
            "No candidate exceeds the dummy baseline. "
            "Investigate before evaluating the test set."
        )

    print("Selected configuration:", winner_name)
    print("Best parameters:", winner_search.best_params_)
    print(f"Training CV macro F1: {winner_search.best_score_:.3f}")

    print("\nValidation classification report:")
    print(classification_report(
        y_valid,
        validation_predictions[winner_name],
        labels=class_labels,
        target_names=class_names,
        digits=3,
        zero_division=0,
    ))

    validation_report = pd.DataFrame(
        classification_report(
            y_valid,
            validation_predictions[winner_name],
            labels=class_labels,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )
    ).T

    validation_report.to_csv(
        model_output_dir / "selected_model_validation_report.csv"
    )

    # Report cell 78
    # Run after model decisions are fixed.
    X_development = pd.concat([X_train, X_valid], axis=0)
    y_development = np.concatenate([y_train, y_valid])

    # Clone removes fitted state but preserves the chosen configuration.
    final_model = clone(winner_search.best_estimator_)
    final_model.fit(X_development, y_development)

    test_predictions = final_model.predict(X_test)

    test_metrics = evaluate_predictions(y_test, test_predictions)
    test_metrics_table = pd.DataFrame(
        [test_metrics],
        index=[winner_name],
    )

    display(test_metrics_table.round(3))

    print(classification_report(
        y_test,
        test_predictions,
        labels=class_labels,
        target_names=class_names,
        digits=3,
        zero_division=0,
    ))

    test_report = pd.DataFrame(
        classification_report(
            y_test,
            test_predictions,
            labels=class_labels,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )
    ).T

    test_metrics_table.to_csv(
        model_output_dir / "final_test_metrics.csv"
    )
    test_report.to_csv(
        model_output_dir / "final_test_classification_report.csv"
    )

    # Save original class names for interpretation.
    prediction_table = pd.DataFrame({
        "Actual": target_encoder.inverse_transform(np.asarray(y_test)),
        "Predicted": target_encoder.inverse_transform(test_predictions),
    }, index=X_test.index)

    prediction_table.index.name = "Original row index"
    prediction_table.to_csv(
        model_output_dir / "final_test_predictions.csv"
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        test_predictions,
        labels=class_labels,
        display_labels=class_names,
        cmap="Blues",
        colorbar=False,
        ax=axes[0],
    )
    axes[0].set_title("Test Confusion Matrix: Counts")

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        test_predictions,
        labels=class_labels,
        display_labels=class_names,
        normalize="true",
        values_format=".2f",
        cmap="Blues",
        colorbar=False,
        ax=axes[1],
    )
    axes[1].set_title("Test Confusion Matrix: Row Proportions")

    fig.tight_layout()
    fig.savefig(
        model_output_dir / "final_test_confusion_matrices.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    # Report cell 82
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from sklearn.inspection import PartialDependenceDisplay
    from sklearn.metrics import accuracy_score, f1_score

    audit_dir = Path("ethical_ai_outputs")
    audit_dir.mkdir(exist_ok=True)

    SEED = 42
    MIN_GROUP_SIZE = 30
    MIN_CLASS_COUNT = 10

    # Preserve positional alignment with y_test.
    X_audit = X_test.copy().reset_index(drop=True)
    y_audit = np.asarray(y_test)
    pred_audit = final_model.predict(X_audit)

    class_names = target_encoder.classes_
    class_labels = target_encoder.transform(class_names)

    assert len(X_audit) == len(y_audit) == len(pred_audit)

    print("Audit observations:", len(X_audit))
    print("Outcomes:", list(class_names))

    # Report cell 84
    dropout_label = int(target_encoder.transform(["Dropout"])[0])

    # Use development data for explanation; no refitting occurs.
    X_explain = pd.concat([X_train, X_valid], axis=0).sample(
        n=min(300, len(X_train) + len(X_valid)),
        random_state=SEED,
    ).copy()

    explanation_features = [
        "Curricular units 2nd sem (approved)",
        "Curricular units 1st sem (approved)",
        "Age at enrollment",
    ]

    fig, axes = plt.subplots(1, 3, figsize=(19, 5))

    PartialDependenceDisplay.from_estimator(
        final_model,
        X_explain,
        features=explanation_features,
        target=dropout_label,
        response_method="predict_proba",
        method="brute",
        kind="both",
        subsample=40,
        grid_resolution=20,
        percentiles=(0.05, 0.95),
        random_state=SEED,
        n_jobs=1,
        ax=axes,
    )

    for ax, feature in zip(axes, explanation_features):
        ax.set_title(feature, wrap=True)
        ax.set_ylabel("Predicted dropout probability")
        ax.set_ylim(0, 1)

    fig.suptitle("Final Model: Average and Individual Responses", y=1.04)
    fig.tight_layout()
    fig.savefig(
        audit_dir / "final_model_pdp_ice.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    # Report cell 87
    def recorded_codes(series):
        """Keep recorded codes; avoid inventing category meanings."""
        return series.astype("string").fillna("Missing")


    groups = pd.DataFrame(index=X_audit.index)

    groups["Gender"] = recorded_codes(X_audit["Gender"])

    groups["Age group"] = (
        pd.cut(
            pd.to_numeric(X_audit["Age at enrollment"], errors="raise"),
            bins=[-np.inf, 20, 25, 35, np.inf],
            labels=["Under 20", "20–24", "25–34", "35+"],
            right=False,
        )
        .astype("string")
        .fillna("Missing")
    )

    for column in ["Scholarship holder", "Debtor", "Tuition fees up to date"]:
        groups[column] = recorded_codes(X_audit[column])

    groups["Gender × Age"] = (
        "Gender " + groups["Gender"] + " / " + groups["Age group"]
    )

    coverage = pd.concat(
        [
            groups[column].value_counts(dropna=False)
            .rename_axis("Group")
            .reset_index(name="Count")
            .assign(Attribute=column)
            for column in groups.columns
        ],
        ignore_index=True,
    )[["Attribute", "Group", "Count"]]

    display(coverage)
    coverage.to_csv(audit_dir / "audit_group_counts.csv", index=False)

    # Report cell 89
    def safe_divide(numerator, denominator):
        return numerator / denominator if denominator > 0 else np.nan


    def wilson_interval(successes, total, z=1.96):
        """Approximate 95% interval for a binomial proportion."""
        if total == 0:
            return np.nan, np.nan

        proportion = successes / total
        denominator = 1 + z**2 / total

        centre = (proportion + z**2 / (2 * total)) / denominator
        half_width = (
            z
            * np.sqrt(
                proportion * (1 - proportion) / total
                + z**2 / (4 * total**2)
            )
            / denominator
        )

        return max(0, centre - half_width), min(1, centre + half_width)


    metric_rows = []

    for attribute in groups.columns:
        for group_name, positions in groups.groupby(
            attribute, sort=True
        ).indices.items():

            actual = y_audit[positions]
            predicted = pred_audit[positions]

            for outcome, label in zip(class_names, class_labels):
                actual_positive = actual == label
                predicted_positive = predicted == label

                tp = int(np.sum(actual_positive & predicted_positive))
                fn = int(np.sum(actual_positive & ~predicted_positive))
                fp = int(np.sum(~actual_positive & predicted_positive))
                tn = int(np.sum(~actual_positive & ~predicted_positive))

                n = len(actual)
                positives = tp + fn
                negatives = fp + tn
                predicted_positives = tp + fp

                selection_low, selection_high = wilson_interval(
                    predicted_positives, n
                )
                tpr_low, tpr_high = wilson_interval(tp, positives)
                fpr_low, fpr_high = wilson_interval(fp, negatives)

                metric_rows.append({
                    "Attribute": attribute,
                    "Group": group_name,
                    "Outcome": outcome,
                    "Count": n,
                    "Actual positives": positives,
                    "Actual negatives": negatives,
                    "Predicted positives": predicted_positives,
                    "TP": tp,
                    "FN": fn,
                    "FP": fp,
                    "TN": tn,
                    "Observed outcome rate": safe_divide(positives, n),
                    "Selection rate": safe_divide(predicted_positives, n),
                    "Selection low": selection_low,
                    "Selection high": selection_high,
                    "TPR": safe_divide(tp, positives),
                    "TPR low": tpr_low,
                    "TPR high": tpr_high,
                    "FPR": safe_divide(fp, negatives),
                    "FPR low": fpr_low,
                    "FPR high": fpr_high,
                    "FNR": safe_divide(fn, positives),
                    "Precision": safe_divide(tp, predicted_positives),
                    "Group accuracy": accuracy_score(actual, predicted),
                    "Group macro F1": f1_score(
                        actual,
                        predicted,
                        labels=class_labels,
                        average="macro",
                        zero_division=0,
                    ),
                    "Small group": n < MIN_GROUP_SIZE,
                    "Sparse outcome support": (
                        positives < MIN_CLASS_COUNT
                        or negatives < MIN_CLASS_COUNT
                    ),
                })

    group_metrics = pd.DataFrame(metric_rows)

    display_columns = [
        "Attribute", "Group", "Outcome", "Count",
        "Actual positives", "Actual negatives",
        "Observed outcome rate", "Selection rate",
        "TPR", "FPR", "FNR", "Precision",
        "Small group", "Sparse outcome support",
    ]

    display(
        group_metrics.loc[
            group_metrics["Outcome"].eq("Dropout"),
            display_columns,
        ].round(3)
    )

    group_metrics.to_csv(
        audit_dir / "fairness_group_metrics_all_outcomes.csv",
        index=False,
    )

    from student_dropout.fairness_summary import summarize
    summarize(group_metrics).to_csv(audit_dir / "fairness_summary_all_outcomes.csv", index=False)

    artifact = {"model": final_model, "target_encoder": target_encoder,
                "columns": list(X_train.columns), "seed": SEED}
    joblib.dump(artifact, ROOT / "models/final_pipeline.joblib")
    versions = {p: importlib.metadata.version(p) for p in
                ["numpy", "pandas", "scipy", "scikit-learn", "joblib"]}
    versions["python"] = platform.python_version()
    (RUN_DIR / "environment.json").write_text(json.dumps(versions, indent=2))
    print("Saved model and outputs to", RUN_DIR)

if __name__ == "__main__":
    main()
