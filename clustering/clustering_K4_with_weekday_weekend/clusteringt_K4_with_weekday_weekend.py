# -*- coding: utf-8 -*-
"""
B同学：建筑能耗聚类分析代码
================================
适用数据：clean_hourly_energy.csv
适用任务：机器学习及在城市大数据分析中的应用 - 建筑能耗聚类部分

本脚本完成：
1. 读取 A 同学处理好的逐小时能耗数据
2. 将“每栋建筑每天24小时单位面积能耗曲线”作为一个聚类样本
3. 使用肘部法则和轮廓系数辅助确定聚类数量
4. 使用 KMeans 聚类
5. 输出聚类中心曲线
6. 分析不同聚类中的工作日/周末、季节、建筑类型差异
7. 补充输出工作日 vs 周末的典型24小时能耗曲线，用于分析二者异同点
8. 保存所有图表和结果表，方便直接放入 PPT

运行方式：
python 02_clustering_B_student.py

作者：B同学聚类分析部分
"""

import os
# 限制线程数，避免部分 Windows / Anaconda 环境下 KMeans 出现 MKL 线程警告或运行变慢
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")


# =========================
# 1. 基本参数设置
# =========================

# 输入文件：A同学已经处理好的数据
INPUT_CSV = "clean_hourly_energy.csv"

# 输出文件夹
OUTPUT_DIR = "outputs_clustering_B_K4_weekdayweekend"
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")
TABLE_DIR = os.path.join(OUTPUT_DIR, "tables")

# 聚类使用的能耗字段
# 推荐使用 load_per_area：单位面积能耗，能够减少建筑面积差异对聚类的影响
LOAD_COL = "load_per_area"

# 测试的聚类数量范围
K_RANGE = range(2, 9)

# 用于评估 k 值的抽样数量。全量样本也可以，但轮廓系数会比较慢。
EVAL_SAMPLE_SIZE = 1000

# 最终聚类数量
# 如果设置为 None，代码会根据轮廓系数自动选择最优 k
# 当前版本固定设置为 4，用于输出 K=4 的完整聚类结果。
# 如果希望完全自动选择，可改成 FINAL_K = None。
FINAL_K = 4

# 随机种子，保证每次运行结果一致
RANDOM_STATE = 42

# 图表文字是否使用英文。
# 如果中文显示成方框，保持 True；英文标题和坐标轴在 PPT 中最稳。
# 如果你在本机确认中文字体正常，也可以改成 False。
USE_ENGLISH_FIG_TEXT = True


# =========================
# 2. 工具函数
# =========================

def ensure_dirs():
    """创建输出文件夹。"""
    os.makedirs(FIGURE_DIR, exist_ok=True)
    os.makedirs(TABLE_DIR, exist_ok=True)


def set_chinese_font():
    """
    设置字体。
    说明：部分 Python / Matplotlib 环境没有中文字体，图中文字会显示成方框。
    因此本脚本默认使用英文图表标题与坐标轴，保证 PPT 中不会乱码。
    如果你想用中文图表文字，可以把 USE_ENGLISH_FIG_TEXT 改成 False，
    并确保电脑中安装了 SimHei 或 Microsoft YaHei 字体。
    """
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
        "Arial Unicode MS", "DejaVu Sans"
    ]
    plt.rcParams["axes.unicode_minus"] = False


def save_current_fig(filename):
    """保存当前图像。"""
    path = os.path.join(FIGURE_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[图表已保存] {path}")


def season_order(series):
    """让季节显示顺序更符合汇报习惯。"""
    order = ["spring", "summer", "autumn", "fall", "winter"]
    existing = [x for x in order if x in series]
    others = [x for x in series if x not in existing]
    return existing + others


# =========================
# 3. 数据读取与检查
# =========================

def load_and_check_data(input_csv):
    """
    读取 clean_hourly_energy.csv，并检查必要字段是否存在。
    """
    if not os.path.exists(input_csv):
        raise FileNotFoundError(
            f"没有找到输入文件：{input_csv}\n"
            f"请把 clean_hourly_energy.csv 放到本脚本同一文件夹下，或者修改 INPUT_CSV 路径。"
        )

    df = pd.read_csv(input_csv)

    required_cols = [
        "building_id", "datetime", "date", "hour", LOAD_COL,
        "category", "month", "season", "is_weekend"
    ]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"数据中缺少必要字段：{missing_cols}")

    df["datetime"] = pd.to_datetime(df["datetime"])
    df["date"] = pd.to_datetime(df["date"]).dt.date

    print("========== 数据基本信息 ==========")
    print(f"数据行数：{len(df)}")
    print(f"建筑数量：{df['building_id'].nunique()}")
    print(f"时间范围：{df['datetime'].min()} 到 {df['datetime'].max()}")
    print(f"缺失值数量：{df.isna().sum().sum()}")
    print("建筑类型数量：")
    print(df[["building_id", "category"]].drop_duplicates()["category"].value_counts())
    print("=================================\n")

    return df


# =========================
# 4. 构造每日24小时聚类样本
# =========================

def build_daily_samples(df):
    """
    将逐小时数据转换为每日24小时曲线。

    输入：
    building_id, date, hour, load_per_area 等逐小时数据

    输出：
    每一行 = 某栋建筑某一天的 24 小时单位面积能耗曲线
    load_0, load_1, ..., load_23 作为聚类特征
    """
    # 透视表：行是 building_id + date，列是 hour，值是 LOAD_COL
    daily_load = df.pivot_table(
        index=["building_id", "date"],
        columns="hour",
        values=LOAD_COL,
        aggfunc="mean"
    )

    # 确保列是 0-23 小时
    daily_load = daily_load.reindex(columns=list(range(24)))

    # 去掉不完整日期，理论上本数据没有不完整日期
    before_drop = len(daily_load)
    daily_load = daily_load.dropna()
    after_drop = len(daily_load)
    if before_drop != after_drop:
        print(f"[提示] 删除了 {before_drop - after_drop} 个不完整的日样本。")

    daily_load.columns = [f"load_{h}" for h in range(24)]
    daily_load = daily_load.reset_index()

    # 每个 building_id + date 对应的辅助信息
    meta_cols = ["building_id", "date", "category", "month", "season", "is_weekend"]
    if "floor" in df.columns:
        meta_cols.append("floor")
    if "area" in df.columns:
        meta_cols.append("area")
    if "total_floor_area" in df.columns:
        meta_cols.append("total_floor_area")

    meta = df[meta_cols].drop_duplicates(subset=["building_id", "date"])

    samples = pd.merge(daily_load, meta, on=["building_id", "date"], how="left")

    # 保存聚类样本表
    sample_path = os.path.join(TABLE_DIR, "clustering_daily_24h_samples.csv")
    samples.to_csv(sample_path, index=False, encoding="utf-8-sig")
    print(f"[表格已保存] {sample_path}")
    print(f"聚类样本数量：{len(samples)}")
    print(f"每个样本维度：24小时能耗曲线\n")

    return samples


# =========================
# 5. 肘部法则与轮廓系数
# =========================

def evaluate_k_values(X_scaled):
    """
    对不同 k 值进行 KMeans 聚类，并计算：
    1. inertia：类内误差平方和，用于肘部法则
    2. silhouette：轮廓系数，越大越好
    3. davies_bouldin：DBI指数，越小越好
    """
    records = []

    # 说明：聚类样本约为 20栋建筑 × 365天 = 7300 个。
    # 轮廓系数需要计算样本间距离，数据稍大时会变慢。
    # 因此这里对 k 值评估进行抽样；最终 KMeans 仍然使用全量样本。
    if len(X_scaled) > EVAL_SAMPLE_SIZE:
        rng = np.random.default_rng(RANDOM_STATE)
        sample_idx = rng.choice(len(X_scaled), size=EVAL_SAMPLE_SIZE, replace=False)
        X_eval = X_scaled[sample_idx]
    else:
        X_eval = X_scaled

    for k in K_RANGE:
        kmeans = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=5
        )
        labels = kmeans.fit_predict(X_eval)

        inertia = kmeans.inertia_
        silhouette = silhouette_score(X_eval, labels)
        dbi = davies_bouldin_score(X_eval, labels)

        records.append({
            "k": k,
            "inertia_SSE": inertia,
            "silhouette_score": silhouette,
            "davies_bouldin_score": dbi
        })

        print(f"k={k}: SSE={inertia:.2f}, silhouette={silhouette:.4f}, DBI={dbi:.4f}")

    eval_df = pd.DataFrame(records)
    eval_path = os.path.join(TABLE_DIR, "kmeans_k_evaluation.csv")
    eval_df.to_csv(eval_path, index=False, encoding="utf-8-sig")
    print(f"[表格已保存] {eval_path}\n")

    # 肘部法则图
    plt.figure(figsize=(7, 5))
    plt.plot(eval_df["k"], eval_df["inertia_SSE"], marker="o")
    if USE_ENGLISH_FIG_TEXT:
        plt.xlabel("Number of clusters k")
        plt.ylabel("SSE / Inertia")
        plt.title("Elbow Method: SSE under Different k Values")
    else:
        plt.xlabel("聚类数量 k")
        plt.ylabel("类内误差平方和 SSE / Inertia")
        plt.title("肘部法则：不同 k 值下的 SSE")
    plt.grid(alpha=0.3)
    save_current_fig("01_elbow_method.png")

    # 轮廓系数图
    plt.figure(figsize=(7, 5))
    plt.plot(eval_df["k"], eval_df["silhouette_score"], marker="o")
    if USE_ENGLISH_FIG_TEXT:
        plt.xlabel("Number of clusters k")
        plt.ylabel("Silhouette Score")
        plt.title("Silhouette Score under Different k Values")
    else:
        plt.xlabel("聚类数量 k")
        plt.ylabel("轮廓系数 Silhouette Score")
        plt.title("不同 k 值下的轮廓系数")
    plt.grid(alpha=0.3)
    save_current_fig("02_silhouette_score.png")

    return eval_df


def choose_final_k(eval_df):
    """
    选择最终 k。
    默认使用轮廓系数最高的 k。
    也可以在脚本顶部手动设置 FINAL_K。
    """
    if FINAL_K is not None:
        final_k = FINAL_K
        print(f"[最终k值] 使用手动设置的 k = {final_k}\n")
    else:
        final_k = int(eval_df.loc[eval_df["silhouette_score"].idxmax(), "k"])
        print(f"[最终k值] 根据轮廓系数自动选择 k = {final_k}\n")

    return final_k


# =========================
# 6. KMeans 聚类与结果输出
# =========================

def run_final_clustering(samples, final_k):
    """
    使用最终 k 进行 KMeans 聚类，并将聚类标签加入样本表。
    """
    feature_cols = [f"load_{h}" for h in range(24)]
    X = samples[feature_cols].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(
        n_clusters=final_k,
        random_state=RANDOM_STATE,
        n_init=10
    )
    labels = kmeans.fit_predict(X_scaled)

    result = samples.copy()
    result["cluster"] = labels

    # 保存聚类结果
    result_path = os.path.join(TABLE_DIR, "clustering_results_with_labels.csv")
    result.to_csv(result_path, index=False, encoding="utf-8-sig")
    print(f"[表格已保存] {result_path}")

    # 输出每类样本数量
    count_df = result["cluster"].value_counts().sort_index().reset_index()
    count_df.columns = ["cluster", "sample_count"]
    count_path = os.path.join(TABLE_DIR, "cluster_sample_count.csv")
    count_df.to_csv(count_path, index=False, encoding="utf-8-sig")
    print(f"[表格已保存] {count_path}")
    print("各聚类样本数量：")
    print(count_df)
    print()

    return result, scaler, kmeans


# =========================
# 7. 聚类中心曲线图
# =========================

def plot_cluster_centers(result):
    """
    绘制聚类中心曲线。
    使用每一类所有样本的 24 小时均值作为中心曲线，更适合解释。
    """
    feature_cols = [f"load_{h}" for h in range(24)]
    hours = np.arange(24)

    center_df = result.groupby("cluster")[feature_cols].mean().reset_index()
    center_path = os.path.join(TABLE_DIR, "cluster_centers_24h_load_per_area.csv")
    center_df.to_csv(center_path, index=False, encoding="utf-8-sig")
    print(f"[表格已保存] {center_path}")

    # 所有中心画在一张图上
    plt.figure(figsize=(9, 5.5))
    for _, row in center_df.iterrows():
        cluster_id = int(row["cluster"])
        values = row[feature_cols].values.astype(float)
        plt.plot(hours, values, marker="o", linewidth=2, label=f"Cluster {cluster_id}")

    if USE_ENGLISH_FIG_TEXT:
        plt.xlabel("Hour of day")
        plt.ylabel("Energy use per area")
        plt.title("24-hour Energy Use Curves of Cluster Centers")
    else:
        plt.xlabel("小时")
        plt.ylabel("单位面积能耗")
        plt.title("不同聚类中心的24小时单位面积能耗曲线")
    plt.xticks(hours)
    plt.grid(alpha=0.3)
    plt.legend()
    save_current_fig("03_cluster_centers_24h_curve.png")

    # 单独画每个聚类中心，方便 PPT 分页展示
    for _, row in center_df.iterrows():
        cluster_id = int(row["cluster"])
        values = row[feature_cols].values.astype(float)

        plt.figure(figsize=(8, 5))
        plt.plot(hours, values, marker="o", linewidth=2)
        if USE_ENGLISH_FIG_TEXT:
            plt.xlabel("Hour of day")
            plt.ylabel("Energy use per area")
            plt.title(f"Typical 24-hour Energy Use Curve of Cluster {cluster_id}")
        else:
            plt.xlabel("小时")
            plt.ylabel("单位面积能耗")
            plt.title(f"Cluster {cluster_id} 的典型24小时能耗曲线")
        plt.xticks(hours)
        plt.grid(alpha=0.3)
        save_current_fig(f"03_cluster_{cluster_id}_center_curve.png")

    return center_df


# =========================
# 8. 聚类解释：工作日/周末、季节、建筑类型
# =========================

def calculate_ratio_table(result, group_col, value_col):
    """
    计算不同 cluster 中某个属性的比例表。
    例如：每个 cluster 中工作日/周末比例、季节比例、建筑类型比例。
    """
    count_table = pd.crosstab(result[group_col], result[value_col])
    ratio_table = pd.crosstab(result[group_col], result[value_col], normalize="index")
    ratio_table = ratio_table * 100
    return count_table, ratio_table


def plot_stacked_ratio(ratio_table, title, ylabel, filename):
    """绘制堆叠柱状图。"""
    ax = ratio_table.plot(kind="bar", stacked=True, figsize=(8, 5.5))
    if USE_ENGLISH_FIG_TEXT:
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Ratio / %")
        ax.set_title(title)
        ax.legend(title="Type", bbox_to_anchor=(1.02, 1), loc="upper left")
    else:
        ax.set_xlabel("聚类类别 Cluster")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(title="类别", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    save_current_fig(filename)


def analyze_cluster_attributes(result):
    """
    分析不同聚类与工作日/周末、季节、建筑类型的关系。
    """
    # 1. 工作日/周末比例
    result = result.copy()
    result["day_type"] = result["is_weekend"].map({0: "weekday", 1: "weekend"})

    count_day, ratio_day = calculate_ratio_table(result, "cluster", "day_type")
    count_day.to_csv(os.path.join(TABLE_DIR, "cluster_day_type_count.csv"), encoding="utf-8-sig")
    ratio_day.to_csv(os.path.join(TABLE_DIR, "cluster_day_type_ratio_percent.csv"), encoding="utf-8-sig")
    plot_stacked_ratio(
        ratio_day,
        "Weekday / Weekend Ratio by Cluster" if USE_ENGLISH_FIG_TEXT else "不同聚类中的工作日/周末比例",
        "比例 / %",
        "04_cluster_weekday_weekend_ratio.png"
    )

    # 2. 季节比例
    count_season, ratio_season = calculate_ratio_table(result, "cluster", "season")
    count_season.to_csv(os.path.join(TABLE_DIR, "cluster_season_count.csv"), encoding="utf-8-sig")
    ratio_season.to_csv(os.path.join(TABLE_DIR, "cluster_season_ratio_percent.csv"), encoding="utf-8-sig")
    plot_stacked_ratio(
        ratio_season,
        "Season Ratio by Cluster" if USE_ENGLISH_FIG_TEXT else "不同聚类中的季节比例",
        "比例 / %",
        "05_cluster_season_ratio.png"
    )

    # 3. 建筑类型比例
    count_cat, ratio_cat = calculate_ratio_table(result, "cluster", "category")
    count_cat.to_csv(os.path.join(TABLE_DIR, "cluster_category_count.csv"), encoding="utf-8-sig")
    ratio_cat.to_csv(os.path.join(TABLE_DIR, "cluster_category_ratio_percent.csv"), encoding="utf-8-sig")
    plot_stacked_ratio(
        ratio_cat,
        "Building Type Ratio by Cluster" if USE_ENGLISH_FIG_TEXT else "不同聚类中的建筑类型比例",
        "比例 / %",
        "06_cluster_building_category_ratio.png"
    )

    print("[表格已保存] 聚类属性比例分析表已保存到 tables 文件夹\n")

    return {
        "day_type_count": count_day,
        "day_type_ratio": ratio_day,
        "season_count": count_season,
        "season_ratio": ratio_season,
        "category_count": count_cat,
        "category_ratio": ratio_cat,
    }




# =========================
# 9. 工作日 / 周末典型能耗曲线对比
# =========================

def plot_weekday_weekend_typical_curves(result):
    """
    补充分析工作日和周末的典型能耗异同点。

    输出两类图：
    1. 全部样本层面的 weekday vs weekend 平均24小时曲线；
    2. 每个 cluster 内部的 weekday vs weekend 平均24小时曲线。

    这部分用于回应作业要求中的：
    “分析工作日和周末典型能耗异同点”。
    """
    feature_cols = [f"load_{h}" for h in range(24)]
    hours = np.arange(24)

    result = result.copy()
    result["day_type"] = result["is_weekend"].map({0: "weekday", 1: "weekend"})

    # ---------- 1. 全部样本：weekday vs weekend ----------
    overall_curve = result.groupby("day_type")[feature_cols].mean().reindex(["weekday", "weekend"])
    overall_curve_path = os.path.join(TABLE_DIR, "overall_weekday_weekend_24h_curve.csv")
    overall_curve.to_csv(overall_curve_path, encoding="utf-8-sig")
    print(f"[表格已保存] {overall_curve_path}")

    plt.figure(figsize=(8.5, 5.2))
    for day_type in overall_curve.index:
        if pd.isna(overall_curve.loc[day_type]).all():
            continue
        plt.plot(
            hours,
            overall_curve.loc[day_type].values.astype(float),
            marker="o",
            linewidth=2,
            label=day_type
        )
    if USE_ENGLISH_FIG_TEXT:
        plt.xlabel("Hour of day")
        plt.ylabel("Energy use per area")
        plt.title("Overall Typical 24-hour Energy Curves: Weekday vs Weekend")
    else:
        plt.xlabel("小时")
        plt.ylabel("单位面积能耗")
        plt.title("整体工作日与周末典型24小时能耗曲线对比")
    plt.xticks(hours)
    plt.grid(alpha=0.3)
    plt.legend()
    save_current_fig("08_overall_weekday_weekend_curve.png")

    # ---------- 2. 每个 cluster 内部：weekday vs weekend ----------
    cluster_day_curve = (
        result
        .groupby(["cluster", "day_type"])[feature_cols]
        .mean()
        .reset_index()
    )
    cluster_day_curve_path = os.path.join(TABLE_DIR, "cluster_weekday_weekend_24h_curve.csv")
    cluster_day_curve.to_csv(cluster_day_curve_path, index=False, encoding="utf-8-sig")
    print(f"[表格已保存] {cluster_day_curve_path}")

    clusters = sorted(result["cluster"].unique())
    for c in clusters:
        sub = cluster_day_curve[cluster_day_curve["cluster"] == c]
        plt.figure(figsize=(8.5, 5.2))
        for day_type in ["weekday", "weekend"]:
            row = sub[sub["day_type"] == day_type]
            if row.empty:
                continue
            values = row[feature_cols].iloc[0].values.astype(float)
            plt.plot(hours, values, marker="o", linewidth=2, label=day_type)
        if USE_ENGLISH_FIG_TEXT:
            plt.xlabel("Hour of day")
            plt.ylabel("Energy use per area")
            plt.title(f"Weekday vs Weekend Typical Curves in Cluster {c}")
        else:
            plt.xlabel("小时")
            plt.ylabel("单位面积能耗")
            plt.title(f"Cluster {c} 内工作日与周末典型能耗曲线对比")
        plt.xticks(hours)
        plt.grid(alpha=0.3)
        plt.legend()
        save_current_fig(f"09_cluster_{c}_weekday_weekend_curve.png")

    # ---------- 3. 同一张图展示所有 cluster 的工作日/周末差异 ----------
    # 使用 2列多行子图，便于 PPT 或附录查看。这里不使用 save_current_fig，因为需要额外控制布局。
    n_clusters = len(clusters)
    ncols = 2
    nrows = int(np.ceil(n_clusters / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4.5 * nrows), sharex=True)
    axes = np.array(axes).reshape(-1)

    for ax, c in zip(axes, clusters):
        sub = cluster_day_curve[cluster_day_curve["cluster"] == c]
        for day_type in ["weekday", "weekend"]:
            row = sub[sub["day_type"] == day_type]
            if row.empty:
                continue
            values = row[feature_cols].iloc[0].values.astype(float)
            ax.plot(hours, values, marker="o", linewidth=2, label=day_type)
        ax.set_title(f"Cluster {c}")
        ax.set_xticks(hours)
        ax.grid(alpha=0.3)
        ax.legend()
        if USE_ENGLISH_FIG_TEXT:
            ax.set_xlabel("Hour of day")
            ax.set_ylabel("Energy use per area")
        else:
            ax.set_xlabel("小时")
            ax.set_ylabel("单位面积能耗")

    for ax in axes[len(clusters):]:
        ax.axis("off")

    if USE_ENGLISH_FIG_TEXT:
        fig.suptitle("Weekday vs Weekend Typical 24-hour Curves by Cluster", fontsize=16)
    else:
        fig.suptitle("各聚类中工作日与周末典型24小时能耗曲线对比", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    combined_path = os.path.join(FIGURE_DIR, "09_cluster_weekday_weekend_curves.png")
    fig.savefig(combined_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[图表已保存] {combined_path}")

    # ---------- 4. 计算工作日/周末差异摘要 ----------
    diff_records = []
    for c in clusters:
        sub = cluster_day_curve[cluster_day_curve["cluster"] == c]
        weekday = sub[sub["day_type"] == "weekday"]
        weekend = sub[sub["day_type"] == "weekend"]
        if weekday.empty or weekend.empty:
            continue
        wkd = weekday[feature_cols].iloc[0].values.astype(float)
        wke = weekend[feature_cols].iloc[0].values.astype(float)
        diff = wkd - wke
        diff_records.append({
            "cluster": c,
            "weekday_mean": wkd.mean(),
            "weekend_mean": wke.mean(),
            "weekday_minus_weekend_mean": diff.mean(),
            "weekday_peak_hour": int(np.argmax(wkd)),
            "weekend_peak_hour": int(np.argmax(wke)),
            "max_abs_difference_hour": int(np.argmax(np.abs(diff))),
            "max_abs_difference_value": float(diff[np.argmax(np.abs(diff))]),
        })

    diff_df = pd.DataFrame(diff_records)
    diff_path = os.path.join(TABLE_DIR, "weekday_weekend_difference_summary.csv")
    diff_df.to_csv(diff_path, index=False, encoding="utf-8-sig")
    print(f"[表格已保存] {diff_path}\n")

    return {
        "overall_weekday_weekend_curve": overall_curve,
        "cluster_weekday_weekend_curve": cluster_day_curve,
        "weekday_weekend_difference_summary": diff_df,
    }


# =========================
# 10. PCA 二维可视化
# =========================

def plot_pca_scatter(result):
    """
    使用 PCA 将24小时能耗曲线降到二维，用于直观展示聚类分布。
    这张图不是必须，但很适合 PPT 展示聚类效果。
    """
    feature_cols = [f"load_{h}" for h in range(24)]
    X = result[feature_cols].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame({
        "PC1": X_pca[:, 0],
        "PC2": X_pca[:, 1],
        "cluster": result["cluster"].values,
        "category": result["category"].values,
        "season": result["season"].values,
        "is_weekend": result["is_weekend"].values,
    })

    pca_path = os.path.join(TABLE_DIR, "pca_2d_cluster_projection.csv")
    pca_df.to_csv(pca_path, index=False, encoding="utf-8-sig")
    print(f"[表格已保存] {pca_path}")

    plt.figure(figsize=(8, 6))
    clusters = sorted(pca_df["cluster"].unique())
    for c in clusters:
        sub = pca_df[pca_df["cluster"] == c]
        plt.scatter(sub["PC1"], sub["PC2"], s=15, alpha=0.65, label=f"Cluster {c}")

    explained = pca.explained_variance_ratio_ * 100
    plt.xlabel(f"PC1 ({explained[0]:.1f}%)")
    plt.ylabel(f"PC2 ({explained[1]:.1f}%)")
    if USE_ENGLISH_FIG_TEXT:
        plt.title("Cluster Distribution in PCA 2D Projection")
    else:
        plt.title("PCA二维投影下的聚类分布")
    plt.legend()
    plt.grid(alpha=0.3)
    save_current_fig("07_pca_cluster_scatter.png")


# =========================
# 11. 自动生成文字分析，方便放进PPT
# =========================

def generate_text_summary(result, center_df, attr_tables, final_k):
    """
    根据结果自动生成一份简短文字总结。
    这部分可以直接复制到 PPT 或演讲稿中，再根据图表具体调整。
    """
    feature_cols = [f"load_{h}" for h in range(24)]
    lines = []
    lines.append("建筑能耗聚类分析总结")
    lines.append("=" * 30)
    lines.append(f"本部分以每栋建筑每天的24小时单位面积能耗曲线作为聚类样本，最终聚类数量 k = {final_k}。")
    lines.append("使用单位面积能耗 load_per_area 是为了减少建筑面积差异对聚类结果的影响，使聚类更关注日内用能曲线形态。")
    lines.append("")

    for _, row in center_df.iterrows():
        c = int(row["cluster"])
        values = row[feature_cols].values.astype(float)
        peak_hour = int(np.argmax(values))
        valley_hour = int(np.argmin(values))
        mean_load = values.mean()
        peak_load = values.max()
        valley_load = values.min()

        lines.append(f"Cluster {c}：")
        lines.append(f"- 样本数量：{(result['cluster'] == c).sum()} 天。")
        lines.append(f"- 平均单位面积能耗：{mean_load:.6f}。")
        lines.append(f"- 峰值出现在 {peak_hour}:00，峰值为 {peak_load:.6f}。")
        lines.append(f"- 谷值出现在 {valley_hour}:00，谷值为 {valley_load:.6f}。")

        # 工作日/周末占比
        day_ratio = attr_tables["day_type_ratio"].loc[c]
        day_desc = ", ".join([f"{idx}: {val:.1f}%" for idx, val in day_ratio.items()])
        lines.append(f"- 工作日/周末占比：{day_desc}。")

        # 季节占比
        season_ratio = attr_tables["season_ratio"].loc[c]
        top_season = season_ratio.idxmax()
        lines.append(f"- 该类中占比最高的季节为 {top_season}，占比 {season_ratio.max():.1f}%。")

        # 建筑类型占比
        cat_ratio = attr_tables["category_ratio"].loc[c]
        top_cat = cat_ratio.idxmax()
        lines.append(f"- 该类中占比最高的建筑类型为 {top_cat}，占比 {cat_ratio.max():.1f}%。")
        lines.append("")

    # 补充整体性的工作日/周末曲线说明
    if "weekday_weekend_difference_summary" in attr_tables:
        ww_diff = attr_tables["weekday_weekend_difference_summary"]
        lines.append("工作日/周末典型曲线补充分析：")
        lines.append("- 已额外输出整体工作日 vs 周末典型24小时曲线，以及各聚类内部工作日 vs 周末曲线。")
        if not ww_diff.empty:
            lines.append("- weekday_weekend_difference_summary.csv 中记录了每个聚类内工作日与周末的平均差值、峰值小时和最大差异小时。")
        lines.append("")

    lines.append("PPT表述建议：")
    lines.append("- 若某类白天能耗明显升高，可以解释为典型办公/高使用强度日间用能模式。")
    lines.append("- 若某类全天能耗较平稳，可以解释为低使用强度或住宅连续用能模式。")
    lines.append("- 若某类在夏季占比较高且下午能耗较大，可以结合空调冷负荷解释。")
    lines.append("- 若某类周末占比较高且白天峰值减弱，可以说明聚类结果能够反映工作日与周末的使用差异。")

    summary_text = "\n".join(lines)
    summary_path = os.path.join(OUTPUT_DIR, "B_student_clustering_text_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text)

    print(f"[文字总结已保存] {summary_path}\n")
    print(summary_text)


# =========================
# 12. 主函数
# =========================

def main():
    ensure_dirs()
    set_chinese_font()

    # 读取数据
    df = load_and_check_data(INPUT_CSV)

    # 构造每日24小时样本
    samples = build_daily_samples(df)

    # 聚类特征：load_0 到 load_23
    feature_cols = [f"load_{h}" for h in range(24)]
    X = samples[feature_cols].values

    # 标准化：避免不同小时或不同建筑量级差异影响距离计算
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 测试不同 k 值
    print("========== 不同 k 值聚类效果评估 ==========")
    eval_df = evaluate_k_values(X_scaled)

    # 选择最终 k
    final_k = choose_final_k(eval_df)

    # 最终聚类
    result, scaler, kmeans = run_final_clustering(samples, final_k)

    # 聚类中心曲线
    center_df = plot_cluster_centers(result)

    # 属性比例分析
    attr_tables = analyze_cluster_attributes(result)

    # 工作日/周末典型曲线对比分析
    weekday_weekend_tables = plot_weekday_weekend_typical_curves(result)
    attr_tables.update(weekday_weekend_tables)

    # PCA 可视化
    plot_pca_scatter(result)

    # 自动生成文字总结
    generate_text_summary(result, center_df, attr_tables, final_k)

    print("========== 全部完成 ==========")
    print(f"所有输出结果保存在：{OUTPUT_DIR}")
    print("建议放入PPT的核心图：")
    print("1. 01_elbow_method.png：肘部法则确定聚类数量")
    print("2. 03_cluster_centers_24h_curve.png：聚类中心曲线")
    print("3. 04_cluster_weekday_weekend_ratio.png：工作日/周末比例")
    print("4. 05_cluster_season_ratio.png：季节比例")
    print("5. 06_cluster_building_category_ratio.png：建筑类型比例")
    print("6. 07_pca_cluster_scatter.png：PCA二维聚类效果展示")
    print("7. 08_overall_weekday_weekend_curve.png：整体工作日/周末典型曲线对比")
    print("8. 09_cluster_weekday_weekend_curves.png：各聚类内部工作日/周末典型曲线对比")


if __name__ == "__main__":
    main()
