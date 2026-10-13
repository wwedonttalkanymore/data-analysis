"""
Netflix titles + Elastic Net regression

Dataset: https://www.kaggle.com/datasets/imtkaggleteam/netflix

This dataset describes titles available on Netflix; its ``rating`` column is
an age/content classification (for example, TV-MA), not a viewer score. This
exercise therefore predicts movie runtime in minutes, not audience ratings.

Usage:
    python 实操演练3.py --data "C:\\path\\to\\NetFlix.csv"
    python 实操演练3.py --data "C:\\path\\to\\netflix_dataset.zip"

If --data is omitted, the script looks for NetFlix.csv or netflix_titles.csv beside this
script and in the current working directory. Results are written to
netflix_elasticnet_results beside the input data file.
"""

from __future__ import annotations

import argparse
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
REQUIRED_COLUMNS = {"type", "duration"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="用 Kaggle Netflix Titles 数据集做 Elastic Net 电影时长分析。"
    )
    parser.add_argument(
        "--data",
        type=Path,
        help="Kaggle 下载的 CSV 文件，或只包含一个目标 CSV 的 ZIP 文件。",
    )
    return parser.parse_args()


def find_dataset(explicit_path: Path | None) -> Path:
    if explicit_path is not None:
        path = explicit_path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"找不到数据文件：{path}")
        return path

    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir / "NetFlix.csv",
        script_dir / "netflix_titles.csv",
        Path.cwd() / "NetFlix.csv",
        Path.cwd() / "netflix_titles.csv",
    ]
    for path in candidates:
        if path.is_file():
            return path.resolve()

    raise FileNotFoundError(
        "没有找到 NetFlix.csv 或 netflix_titles.csv。请从 Kaggle 下载数据后，运行：\n"
        '  python 实操演练3.py --data "数据文件的完整路径.csv"\n'
        "也可以将 CSV 放在本脚本同一目录下。"
    )


def read_dataset(path: Path) -> pd.DataFrame:
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            csv_files = [
                name for name in archive.namelist()
                if name.lower().endswith(".csv") and not name.endswith("/")
            ]
            if not csv_files:
                raise ValueError(f"压缩包内没有 CSV 文件：{path}")
            preferred = [
                name for name in csv_files
                if Path(name).name.lower() == "netflix_titles.csv"
            ]
            if len(preferred) == 1:
                selected = preferred[0]
            elif len(csv_files) == 1:
                selected = csv_files[0]
            else:
                raise ValueError(
                    "压缩包内有多个 CSV，无法确定要分析哪一个：\n  "
                    + "\n  ".join(csv_files)
                    + "\n请解压后通过 --data 指定 netflix_titles.csv。"
                )
            with archive.open(selected) as csv_stream:
                return pd.read_csv(csv_stream)
    return pd.read_csv(path)


def print_data_overview(data: pd.DataFrame) -> None:
    print("\n=== 数据概览 ===")
    print(f"数据行数：{len(data):,}")
    print(f"字段数量：{len(data.columns)}")
    print("\n字段名：")
    print(", ".join(map(str, data.columns)))
    print("\n字段缺失比例（前 12 项）：")
    missing = data.isna().mean().sort_values(ascending=False).head(12)
    print((missing * 100).round(1).astype(str) + "%")

    if "type" in data.columns:
        print("\n内容类型：")
        print(data["type"].value_counts(dropna=False).to_string())
    if "rating" in data.columns:
        print(
            "\n提示：数据中的 rating 是内容分级（如 TV-MA），"
            "不是观众评分。"
        )


def prepare_movies(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    if "listed_in" not in data.columns and "genres" in data.columns:
        data["listed_in"] = data["genres"]

    missing = REQUIRED_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(
            "这个文件不像 Netflix Titles 目录数据，缺少必要字段："
            + ", ".join(sorted(missing))
            + "。本练习需要包含 type 和 duration 的 Netflix 标题目录 CSV。"
        )

    movies = data.loc[
        data["type"].astype("string").str.strip().str.casefold().eq("movie")
    ].copy()
    duration_text = movies["duration"].astype("string")
    movies["target_minutes"] = pd.to_numeric(
        movies["duration"], errors="coerce"
    )
    text_durations = movies["target_minutes"].isna()
    if text_durations.any():
        movies.loc[text_durations, "target_minutes"] = pd.to_numeric(
            duration_text[text_durations].str.extract(
                r"^\s*(\d+)\s*min\b", flags=re.IGNORECASE
            )[0],
            errors="coerce",
        )
    movies = movies.dropna(subset=["target_minutes"]).copy()
    movies["target_minutes"] = movies["target_minutes"].astype(float)

    if movies.empty:
        raise ValueError(
            "没有解析出电影时长。请检查 duration 是否为分钟数，"
            "或采用类似 '90 min' 的格式。"
        )

    for column in ("release_year", "date_added", "rating", "country", "listed_in"):
        if column not in movies.columns:
            movies[column] = np.nan

    movies["release_year"] = pd.to_numeric(
        movies["release_year"], errors="coerce"
    )
    added_dates = pd.to_datetime(
        movies["date_added"], format="%d-%b-%y", errors="coerce"
    )
    for date_format in ("%B %d, %Y", "%b %d, %Y"):
        missing_dates = added_dates.isna() & movies["date_added"].notna()
        if not missing_dates.any():
            break
        added_dates.loc[missing_dates] = pd.to_datetime(
            movies.loc[missing_dates, "date_added"],
            format=date_format,
            errors="coerce",
        )
    movies["added_year"] = added_dates.dt.year

    for column in ("rating", "country", "listed_in"):
        movies[column] = (
            movies[column]
            .astype("string")
            .str.strip()
            .replace("", pd.NA)
            .fillna("Unknown")
            .astype(str)
        )
    movies["listed_in"] = movies["listed_in"].str.replace(
        ",", " ", regex=False
    )
    return movies


def build_model() -> Pipeline:
    numeric_features = ["release_year", "added_year"]
    categorical_features = ["rating", "country"]

    preprocess = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=10,
                ),
                categorical_features,
            ),
            (
                "genres",
                CountVectorizer(
                    binary=True,
                    token_pattern=r"(?u)\b\w[\w-]*\b",
                    min_df=5,
                ),
                "listed_in",
            ),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            (
                "model",
                ElasticNet(
                    max_iter=20_000,
                    tol=1e-4,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def run_analysis(movies: pd.DataFrame, output_dir: Path) -> None:
    feature_columns = [
        "release_year", "added_year", "rating", "country", "listed_in"
    ]
    x = movies[feature_columns]
    y = movies["target_minutes"]

    if len(movies) < 10:
        raise ValueError(
            f"可用电影只有 {len(movies)} 条，至少需要 10 条才能进行训练和测试划分。"
        )

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )
    cv_splits = min(5, len(x_train))
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)

    search = GridSearchCV(
        estimator=build_model(),
        param_grid={
            "model__alpha": [0.01, 0.1, 1.0, 10.0],
            "model__l1_ratio": [0.1, 0.5, 0.9, 1.0],
        },
        scoring={
            "rmse": "neg_root_mean_squared_error",
            "mae": "neg_mean_absolute_error",
        },
        refit="rmse",
        cv=cv,
        n_jobs=-1,
        error_score="raise",
    )

    print("\n=== 分析设定 ===")
    print("研究问题：电影类别、内容分级、国家及年份能否预测电影时长？")
    print(f"可用电影：{len(movies):,}")
    print(f"训练集：{len(x_train):,}；测试集：{len(x_test):,}")
    print("目标变量：duration 中的电影分钟数")
    print("特征：上映年份、上线年份、内容分级、国家、类型")
    print("模型：Elastic Net；5 折交叉验证（数据较少时自动减少折数）")

    search.fit(x_train, y_train)
    predictions = search.predict(x_test)

    model_rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    model_mae = float(mean_absolute_error(y_test, predictions))
    model_r2 = float(r2_score(y_test, predictions))
    baseline_value = float(y_train.median())
    baseline_predictions = np.full(len(y_test), baseline_value)
    baseline_rmse = float(np.sqrt(mean_squared_error(y_test, baseline_predictions)))
    baseline_mae = float(mean_absolute_error(y_test, baseline_predictions))
    baseline_r2 = float(r2_score(y_test, baseline_predictions))

    metrics = pd.DataFrame(
        [
            {
                "model": "Elastic Net",
                "test_RMSE_minutes": model_rmse,
                "test_MAE_minutes": model_mae,
                "test_R2": model_r2,
                "selected_alpha": search.best_params_["model__alpha"],
                "selected_l1_ratio": search.best_params_["model__l1_ratio"],
            },
            {
                "model": "Training median baseline",
                "test_RMSE_minutes": baseline_rmse,
                "test_MAE_minutes": baseline_mae,
                "test_R2": baseline_r2,
                "selected_alpha": np.nan,
                "selected_l1_ratio": np.nan,
            },
        ]
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "model_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        {
            "actual_minutes": y_test.to_numpy(),
            "predicted_minutes": predictions,
            "absolute_error_minutes": np.abs(y_test.to_numpy() - predictions),
        }
    ).to_csv(
        output_dir / "test_predictions.csv", index=False, encoding="utf-8-sig"
    )

    fitted_preprocess = search.best_estimator_.named_steps["preprocess"]
    fitted_model = search.best_estimator_.named_steps["model"]
    feature_names = fitted_preprocess.get_feature_names_out()
    coefficients = pd.DataFrame(
        {"feature": feature_names, "coefficient": fitted_model.coef_}
    )
    coefficients["abs_coefficient"] = coefficients["coefficient"].abs()
    coefficients.sort_values(
        "abs_coefficient", ascending=False
    ).to_csv(
        output_dir / "elasticnet_coefficients.csv",
        index=False,
        encoding="utf-8-sig",
    )

    search_results = pd.DataFrame(search.cv_results_)
    search_results[
        [
            "param_model__alpha",
            "param_model__l1_ratio",
            "mean_test_rmse",
            "mean_test_mae",
            "rank_test_rmse",
        ]
    ].assign(
        mean_test_rmse=lambda frame: -frame["mean_test_rmse"],
        mean_test_mae=lambda frame: -frame["mean_test_mae"],
    ).sort_values("rank_test_rmse").to_csv(
        output_dir / "cross_validation_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n=== 测试集结果 ===")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(
        "\n指标说明：RMSE / MAE 越低越好；R² 越高越好。"
        "Elastic Net 优于中位数基线时，才说明这些特征有预测价值。"
    )
    print(
        "\n注意：这是相关性预测练习，不是因果分析；"
        "随机划分的结果不代表对未来影片的时间外推能力。"
    )
    print(f"\n结果文件已保存到：{output_dir}")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    args = parse_args()
    try:
        data_path = find_dataset(args.data)
        print(f"读取数据：{data_path}")
        data = read_dataset(data_path)
        print_data_overview(data)
        movies = prepare_movies(data)

        print("\n=== 电影时长描述统计（分钟）===")
        print(movies["target_minutes"].describe().round(1).to_string())

        output_dir = data_path.parent / "netflix_elasticnet_results"
        run_analysis(movies, output_dir)
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as error:
        print(f"\n无法完成分析：{error}", file=sys.stderr)
        return 1
    except (OSError, zipfile.BadZipFile) as error:
        print(f"\n读取或写入文件失败：{error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())