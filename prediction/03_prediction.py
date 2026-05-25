"""
03_prediction.py
建筑能耗短期负荷预测模型

任务描述：
    用前 n 小时能耗（滞后特征）+ 天气特征 + 时间特征 + 建筑信息
    预测当前小时的总能耗（total_load），输出 RMSE、R2，并绘图。

模型选择：Random Forest（主模型）
比较窗口：lag_6、lag_12、lag_24（选 lag_24 作为最终模型）

输出：
    prediction_metrics.csv                  —— RMSE / R2 汇总表
    figures/prediction_vs_actual.png        —— 真实值与预测值对比曲线
    figures/feature_importance.png          —— 特征重要性柱状图
    figures/lag_window_comparison.png       —— 不同滞后窗口的 RMSE / R2 对比
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# =========================================================
# 路径配置
# =========================================================

BASE_DIR   = Path(__file__).resolve().parent
FIG_DIR    = BASE_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# A 同学预处理数据所在目录
DATA_DIR = BASE_DIR.parent / "final_assignment" / "final_assignment" / "outputs"
HOURLY_DATA_PATH = DATA_DIR / "clean_hourly_energy.csv"

# 绘图全局风格
plt.rcParams.update({
    "figure.dpi":     150,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "font.size":      11,
})

PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]


# =========================================================
# 1. 读取预处理好的逐小时数据
# =========================================================

def load_hourly(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["datetime"])
    df = df.sort_values(["building_id", "datetime"]).reset_index(drop=True)
    return df


# =========================================================
# 2. 构造特征
# =========================================================

def build_features(df: pd.DataFrame, lag_hours: int) -> pd.DataFrame:
    """
    为每栋建筑独立构造滞后特征，避免跨建筑污染。

    最终特征集：
      - lag_1 ~ lag_{lag_hours}：前 lag_hours 个小时的 total_load
      - hour, month, is_weekend（时间特征）
      - dry_bulb_temperature, relative_humidity,
        global_horizontal_radiation, wind_speed（天气特征）
      - floor, area, total_floor_area（建筑信息）
      - category_enc（建筑类型编码）
    """

    records = []

    le = LabelEncoder()
    le.fit(df["category"].astype(str))

    for bid, group in df.groupby("building_id"):
        g = group.copy().sort_values("datetime").reset_index(drop=True)

        # 滞后特征
        for lag in range(1, lag_hours + 1):
            g[f"lag_{lag}"] = g["total_load"].shift(lag)

        # 删掉前 lag_hours 行（缺少历史数据）
        g = g.dropna(subset=[f"lag_{lag_hours}"]).copy()

        # 建筑类型编码
        g["category_enc"] = le.transform(g["category"].astype(str))

        records.append(g)

    data = pd.concat(records, ignore_index=True)
    return data, le


def get_feature_cols(lag_hours: int) -> list:
    lag_cols   = [f"lag_{i}" for i in range(1, lag_hours + 1)]
    time_cols  = ["hour", "month", "is_weekend"]
    weather_cols = [
        "dry_bulb_temperature",
        "relative_humidity",
        "global_horizontal_radiation",
        "wind_speed",
    ]
    building_cols = ["floor", "area", "total_floor_area", "category_enc"]
    return lag_cols + time_cols + weather_cols + building_cols


# =========================================================
# 3. 训练 / 评估
# =========================================================

def train_evaluate(
    data: pd.DataFrame,
    lag_hours: int,
    n_estimators: int = 200,
    random_state: int = 42,
) -> dict:
    """
    按时间节点划分训练集（前 80% 时间）和测试集（后 20% 时间）。
    对所有 20 栋建筑统一使用同一个时间切割点，保证测试集覆盖所有建筑。
    2021 年共 365 天，80% 约为前 292 天，切割点为 2021-10-20。
    """

    feature_cols = get_feature_cols(lag_hours)
    target_col   = "total_load"

    # 按时间切割：所有建筑统一以 2021-10-20 为界
    split_date = pd.Timestamp("2021-10-20")
    train = data[data["datetime"] < split_date]
    test  = data[data["datetime"] >= split_date]

    X_train = train[feature_cols]
    y_train = train[target_col]
    X_test  = test[feature_cols]
    y_test  = test[target_col]

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_features="sqrt",
        n_jobs=-1,
        random_state=random_state,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    print(f"  lag={lag_hours:2d}h  RMSE={rmse:.4f}  R2={r2:.4f}")

    return {
        "lag_hours":     lag_hours,
        "rmse":          rmse,
        "r2":            r2,
        "model":         model,
        "y_test":        y_test.values,
        "y_pred":        y_pred,
        "test_datetime": test["datetime"].values,
        "feature_cols":  feature_cols,
        "test_building": test["building_id"].values,
    }


# =========================================================
# 4. 绘图函数
# =========================================================

def plot_lag_comparison(results: list, save_path: Path) -> None:
    """不同滞后窗口 RMSE / R2 对比柱状图"""

    lags  = [r["lag_hours"] for r in results]
    rmses = [r["rmse"]      for r in results]
    r2s   = [r["r2"]        for r in results]

    x = np.arange(len(lags))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    bars1 = ax1.bar(x, rmses, width, color=PALETTE[:len(lags)])
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"lag={l}h" for l in lags])
    ax1.set_ylabel("RMSE")
    ax1.set_title("Lag Window Comparison — RMSE")
    for bar, v in zip(bars1, rmses):
        ax1.text(bar.get_x() + bar.get_width() / 2, v * 1.01,
                 f"{v:.4f}", ha="center", va="bottom", fontsize=9)

    bars2 = ax2.bar(x, r2s, width, color=PALETTE[:len(lags)])
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"lag={l}h" for l in lags])
    ax2.set_ylabel("R2")
    ax2.set_title("Lag Window Comparison - R2")
    ax2.set_ylim(0, 1.05)
    for bar, v in zip(bars2, r2s):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                 f"{v:.4f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Random Forest — Lag Window Comparison", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  图已保存：{save_path.name}")


def plot_prediction_vs_actual(result: dict, save_path: Path,
                               building_id: int = None,
                               n_days: int = 14) -> None:
    """
    选取一栋建筑最后 n_days 天的真实值与预测值对比折线图。
    若 building_id 为 None，自动选择测试集中均值最大的建筑。
    """

    buildings = result["test_building"]
    y_test    = result["y_test"]

    if building_id is None:
        # 找到测试集中均值负荷最大的建筑
        unique_bids = np.unique(buildings)
        mean_loads  = {bid: y_test[buildings == bid].mean() for bid in unique_bids}
        building_id = max(mean_loads, key=mean_loads.get)
        print(f"  自动选择建筑 building_{building_id} 绘制对比图（平均负荷最大）")

    mask = buildings == building_id
    dt   = pd.to_datetime(result["test_datetime"][mask])
    y_t  = y_test[mask]
    y_p  = result["y_pred"][mask]

    # 只取最后 n_days * 24 个点，方便观察细节
    n_pts = n_days * 24
    dt  = dt[-n_pts:]
    y_t = y_t[-n_pts:]
    y_p = y_p[-n_pts:]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(dt, y_t, label="Actual",    color=PALETTE[0], linewidth=1.2)
    ax.plot(dt, y_p, label="Predicted", color=PALETTE[1], linewidth=1.0,
            linestyle="--", alpha=0.85)
    ax.set_xlabel("Datetime")
    ax.set_ylabel("Total Load (kWh)")
    ax.set_title(
        f"Building {building_id} - Actual vs Predicted "
        f"(lag={result['lag_hours']}h, last {n_days} days)\n"
        f"RMSE={result['rmse']:.4f}  R2={result['r2']:.4f}"
    )
    ax.legend(loc="upper right")
    fig.autofmt_xdate(rotation=25)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  图已保存：{save_path.name}")


def plot_scatter(result: dict, save_path: Path) -> None:
    """真实值 vs 预测值散点图（全测试集，抽样 5000 点）"""

    y_t = result["y_test"]
    y_p = result["y_pred"]

    # 最多抽 5000 点
    np.random.seed(42)
    idx = np.random.choice(len(y_t), min(5000, len(y_t)), replace=False)
    y_t_s = y_t[idx]
    y_p_s = y_p[idx]

    vmin = min(y_t_s.min(), y_p_s.min())
    vmax = max(y_t_s.max(), y_p_s.max())

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_t_s, y_p_s, s=6, alpha=0.4, color=PALETTE[0])
    ax.plot([vmin, vmax], [vmin, vmax], "r--", linewidth=1.2, label="y=x")
    ax.set_xlabel("Actual Total Load (kWh)")
    ax.set_ylabel("Predicted Total Load (kWh)")
    ax.set_title(
        f"Actual vs Predicted Scatter (lag={result['lag_hours']}h)\n"
        f"RMSE={result['rmse']:.4f}  R2={result['r2']:.4f}"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  图已保存：{save_path.name}")


def plot_feature_importance(result: dict, save_path: Path, top_n: int = 20) -> None:
    """特征重要性横向柱状图"""

    model         = result["model"]
    feature_cols  = result["feature_cols"]
    importances   = model.feature_importances_

    feat_df = pd.DataFrame({
        "feature":    feature_cols,
        "importance": importances,
    }).sort_values("importance", ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(8, 0.4 * top_n + 1))
    bars = ax.barh(feat_df["feature"], feat_df["importance"], color=PALETTE[0])
    ax.set_xlabel("Feature Importance (Mean Decrease Impurity)")
    ax.set_title(f"Top-{top_n} Feature Importance (Random Forest, lag={result['lag_hours']}h)")
    for bar, v in zip(bars, feat_df["importance"]):
        ax.text(v + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{v:.4f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  图已保存：{save_path.name}")


# =========================================================
# 5. 主程序
# =========================================================

def main():
    print("=" * 60)
    print("03_prediction.py -- 建筑能耗预测模型")
    print("=" * 60)

    # ── 读取数据 ──────────────────────────────────────────
    print("\n[1/5] 读取逐小时数据...")
    df = load_hourly(HOURLY_DATA_PATH)
    print(f"  数据维度：{df.shape}")
    print(f"  建筑数量：{df['building_id'].nunique()}")
    print(f"  时间范围：{df['datetime'].min()} → {df['datetime'].max()}")

    # ── 比较不同滞后窗口 ──────────────────────────────────
    lag_windows = [6, 12, 24]
    results_all  = []

    print("\n[2/5] 训练与评估（三个滞后窗口）...")

    for lag in lag_windows:
        print(f"\n  构造特征（lag={lag}h）...")
        data, le = build_features(df, lag_hours=lag)
        res = train_evaluate(data, lag_hours=lag)
        res["label_encoder"] = le
        results_all.append(res)

    # ── 最终模型：lag_24 ──────────────────────────────────
    best_result = results_all[-1]      # lag=24

    # ── 保存指标 ──────────────────────────────────────────
    print("\n[3/5] 保存评估指标...")
    metrics_df = pd.DataFrame([
        {
            "model":      "Random Forest",
            "lag_hours":  r["lag_hours"],
            "rmse":       round(r["rmse"], 6),
            "r2":         round(r["r2"],   6),
        }
        for r in results_all
    ])
    metrics_path = BASE_DIR / "prediction_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    print(f"  指标已保存：{metrics_path.name}")
    print(metrics_df.to_string(index=False))

    # ── 绘图 ──────────────────────────────────────────────
    print("\n[4/5] 绘制图表...")

    # 图1：滞后窗口对比
    plot_lag_comparison(
        results_all,
        FIG_DIR / "lag_window_comparison.png"
    )

    # 图2：时间序列对比折线图（自动选均值最大建筑，最后 14 天）
    plot_prediction_vs_actual(
        best_result,
        FIG_DIR / "prediction_vs_actual.png",
        building_id=None,
        n_days=14,
    )

    # 图3：全测试集散点图
    plot_scatter(
        best_result,
        FIG_DIR / "prediction_scatter.png"
    )

    # 图4：特征重要性
    plot_feature_importance(
        best_result,
        FIG_DIR / "feature_importance.png",
        top_n=20,
    )

    # ── 汇总打印 ──────────────────────────────────────────
    print("\n[5/5] 汇总结果")
    print(f"\n  最终模型：Random Forest，滞后窗口 = 24 小时")
    print(f"  RMSE = {best_result['rmse']:.6f}")
    print(f"  R2   = {best_result['r2']:.6f}")
    print("\n  输出文件：")
    print(f"    {metrics_path}")
    print(f"    {FIG_DIR / 'lag_window_comparison.png'}")
    print(f"    {FIG_DIR / 'prediction_vs_actual.png'}")
    print(f"    {FIG_DIR / 'prediction_scatter.png'}")
    print(f"    {FIG_DIR / 'feature_importance.png'}")
    print(f"\n  输入数据：{HOURLY_DATA_PATH}")
    print("\n预测任务完成。")


if __name__ == "__main__":
    main()
