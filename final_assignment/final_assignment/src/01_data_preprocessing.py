from pathlib import Path
import pandas as pd
import numpy as np


# =========================================================
# 1. 相对路径设置
# =========================================================

# 当前代码文件所在文件夹
SRC_DIR = Path(__file__).resolve().parent

# src 的上一级
BASE_DIR = SRC_DIR.parent

# 数据文件夹：final_assignment/data
DATA_DIR = BASE_DIR / "data"

# 天气文件夹：final_assignment/data/weather
WEATHER_DIR = DATA_DIR / "weather"

# 输出文件夹：final_assignment/outputs
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# 建筑信息表
BUILDING_INFO_PATH = DATA_DIR / "building_info.csv"

# 自动寻找 weather 文件夹中的 epw 文件
EPW_FILES = list(WEATHER_DIR.glob("*.epw"))

if len(EPW_FILES) == 0:
    raise FileNotFoundError(f"没有在这个文件夹中找到 .epw 文件：{WEATHER_DIR}")

EPW_PATH = EPW_FILES[0]


# =========================================================
# 2. 读取建筑基本信息
# =========================================================

def load_building_info(path: Path) -> pd.DataFrame:
    """
    读取 building_info.csv。

    原始字段：
    index：建筑编号
    category：建筑类型
    floor：楼层数
    area：建筑投影面积
    """

    df = pd.read_csv(path)

    # 清理列名，防止列名中有隐藏空格
    df.columns = [col.strip() for col in df.columns]

    required_cols = ["index", "category", "floor", "area"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"building_info.csv 缺少必要字段：{col}")

    # 将 index 改成 building_id，方便后面和 building_1.csv 等文件对应
    df = df.rename(columns={"index": "building_id"})

    # 清理建筑类型字段
    df["category"] = df["category"].astype(str).str.strip()

    # 楼层数、投影面积转为数值
    df["floor"] = pd.to_numeric(df["floor"], errors="coerce")
    df["area"] = pd.to_numeric(df["area"], errors="coerce")

    # 估算总建筑面积：投影面积 × 楼层数
    # 这个变量后续可以用于预测模型
    df["total_floor_area"] = df["floor"] * df["area"]

    return df


# =========================================================
# 3. 读取 EPW 天气数据
# =========================================================

def load_epw_weather(path: Path) -> pd.DataFrame:
    """
    读取 EPW 天气文件。

    EPW 文件前 8 行是说明信息。
    从第 9 行开始是逐小时天气数据。

    提取几个常用变量：
    - dry_bulb_temperature：干球温度
    - relative_humidity：相对湿度
    - global_horizontal_radiation：全球水平太阳辐射
    - direct_normal_radiation：直接法向太阳辐射
    - diffuse_horizontal_radiation：散射水平太阳辐射
    - wind_speed：风速
    """

    epw_columns = [
        "year",
        "month",
        "day",
        "hour",
        "minute",
        "data_source",
        "dry_bulb_temperature",
        "dew_point_temperature",
        "relative_humidity",
        "atmospheric_station_pressure",
        "extraterrestrial_horizontal_radiation",
        "extraterrestrial_direct_normal_radiation",
        "horizontal_infrared_radiation_intensity",
        "global_horizontal_radiation",
        "direct_normal_radiation",
        "diffuse_horizontal_radiation",
        "global_horizontal_illuminance",
        "direct_normal_illuminance",
        "diffuse_horizontal_illuminance",
        "zenith_luminance",
        "wind_direction",
        "wind_speed",
        "total_sky_cover",
        "opaque_sky_cover",
        "visibility",
        "ceiling_height",
        "present_weather_observation",
        "present_weather_codes",
        "precipitable_water",
        "aerosol_optical_depth",
        "snow_depth",
        "days_since_last_snowfall",
        "albedo",
        "liquid_precipitation_depth",
        "liquid_precipitation_quantity"
    ]

    weather = pd.read_csv(
        path,
        skiprows=8,
        header=None,
        names=epw_columns
    )

    # 只保留 8760 小时
    weather = weather.iloc[:8760].copy()

    # 建立 2021 年逐小时时间索引
    # 注意：TMY 天气文件中的年份不一定是真实年份，所以这里统一映射到 2021 年
    weather["datetime"] = pd.date_range(
        start="2021-01-01 00:00:00",
        periods=len(weather),
        freq="h"
    )

    keep_cols = [
        "datetime",
        "dry_bulb_temperature",
        "relative_humidity",
        "global_horizontal_radiation",
        "direct_normal_radiation",
        "diffuse_horizontal_radiation",
        "wind_speed"
    ]

    weather = weather[keep_cols]

    # 天气变量转成数值
    for col in keep_cols:
        if col != "datetime":
            weather[col] = pd.to_numeric(weather[col], errors="coerce")

    return weather


# =========================================================
# 4. 读取单栋建筑能耗数据
# =========================================================

def load_one_building_energy(path: Path, building_id: int) -> pd.DataFrame:
    """
    读取单栋建筑的逐小时能耗数据。

    每个 building_n.csv 里包含多个分项能耗，例如：
    cooling, heating, lighting, electric_hot_water,
    gas_equipment, process, fan_electric, pump_electric。

    这里将所有分项能耗相加，得到 total_load。
    """

    df = pd.read_csv(path)

    # 清理列名
    df.columns = [col.strip() for col in df.columns]

    # 只保留前 8760 行
    df = df.iloc[:8760].copy()

    # 建立时间索引
    df["datetime"] = pd.date_range(
        start="2021-01-01 00:00:00",
        periods=len(df),
        freq="h"
    )

    # 添加建筑编号
    df["building_id"] = building_id

    # 除 datetime 和 building_id 之外，其余列都视为分项能耗
    energy_cols = [
        col for col in df.columns
        if col not in ["datetime", "building_id"]
    ]

    # 所有能耗列转为数值
    for col in energy_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 计算总能耗
    df["total_load"] = df[energy_cols].sum(axis=1)

    return df


# =========================================================
# 5. 读取 20 栋建筑能耗数据
# =========================================================

def load_all_building_energy(data_dir: Path) -> pd.DataFrame:
    """
    读取 building_1.csv 到 building_20.csv。
    最后合并成长表。
    """

    all_building_data = []

    for i in range(1, 21):
        file_path = data_dir / f"building_{i}.csv"

        if not file_path.exists():
            raise FileNotFoundError(f"没有找到建筑能耗文件：{file_path}")

        one_building = load_one_building_energy(file_path, i)
        all_building_data.append(one_building)

        print(f"已读取：building_{i}.csv，数据维度：{one_building.shape}")

    energy = pd.concat(all_building_data, ignore_index=True)

    return energy


# =========================================================
# 6. 构造时间特征
# =========================================================

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    增加时间特征。

    hour：一天中的小时，0-23
    dayofweek：一周中的第几天，周一=0，周日=6
    month：月份
    date：日期
    is_weekend：是否周末
    season：季节
    """

    df = df.copy()

    df["hour"] = df["datetime"].dt.hour
    df["dayofweek"] = df["datetime"].dt.dayofweek
    df["month"] = df["datetime"].dt.month
    df["date"] = df["datetime"].dt.date

    # 周六、周日记为周末
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

    def get_season(month: int) -> str:
        if month in [3, 4, 5]:
            return "spring"
        elif month in [6, 7, 8]:
            return "summer"
        elif month in [9, 10, 11]:
            return "autumn"
        else:
            return "winter"

    df["season"] = df["month"].apply(get_season)

    return df


# =========================================================
# 7. 生成聚类用的每日 24 小时矩阵
# =========================================================

def make_daily_load_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    将逐小时数据转换成每日 24 小时能耗曲线。

    原始结构：
    一行 = 某栋建筑某一小时

    转换后：
    一行 = 某栋建筑某一天

    输出字段类似：
    building_id, date, category, floor, area,
    total_floor_area, season, is_weekend,
    load_0, load_1, ..., load_23
    """

    daily = df.pivot_table(
        index=["building_id", "date"],
        columns="hour",
        values="total_load",
        aggfunc="mean"
    ).reset_index()

    # 将 0,1,2...23 改成 load_0, load_1, ..., load_23
    new_columns = []

    for col in daily.columns:
        if isinstance(col, int):
            new_columns.append(f"load_{col}")
        else:
            new_columns.append(col)

    daily.columns = new_columns

    # 每栋建筑每天只需要保留一条元信息
    meta_cols = [
        "building_id",
        "date",
        "category",
        "floor",
        "area",
        "total_floor_area",
        "season",
        "is_weekend"
    ]

    meta = df[meta_cols].drop_duplicates(subset=["building_id", "date"])

    daily = daily.merge(
        meta,
        on=["building_id", "date"],
        how="left"
    )

    # 调整列顺序
    load_cols = [f"load_{i}" for i in range(24)]

    daily = daily[
        [
            "building_id",
            "date",
            "category",
            "floor",
            "area",
            "total_floor_area",
            "season",
            "is_weekend"
        ]
        + load_cols
    ]

    return daily


# =========================================================
# 8. 简单检查数据质量
# =========================================================

def check_data_quality(hourly_df: pd.DataFrame, daily_df: pd.DataFrame) -> None:
    """
    输出一些简单检查信息，方便确认数据是否正常。
    """

    print("\n========== 数据质量检查 ==========")

    print("逐小时数据维度：", hourly_df.shape)
    print("每日矩阵数据维度：", daily_df.shape)

    print("\n建筑数量：", hourly_df["building_id"].nunique())
    print("时间起点：", hourly_df["datetime"].min())
    print("时间终点：", hourly_df["datetime"].max())

    print("\n每栋建筑的小时数据量：")
    print(hourly_df.groupby("building_id").size())

    print("\n建筑类型分布：")
    print(hourly_df[["building_id", "category"]].drop_duplicates()["category"].value_counts())

    print("\n主要字段缺失值数量：")
    check_cols = [
        "total_load",
        "category",
        "floor",
        "area",
        "total_floor_area",
        "dry_bulb_temperature",
        "relative_humidity",
        "global_horizontal_radiation",
        "wind_speed"
    ]

    existing_check_cols = [col for col in check_cols if col in hourly_df.columns]
    print(hourly_df[existing_check_cols].isna().sum())

    print("========== 检查结束 ==========\n")


# =========================================================
# 9. 主程序
# =========================================================

def main():
    print("项目根目录：", BASE_DIR)
    print("数据文件夹：", DATA_DIR)
    print("天气文件夹：", WEATHER_DIR)
    print("输出文件夹：", OUTPUT_DIR)
    print("使用的 EPW 文件：", EPW_PATH)

    print("\n开始读取建筑信息...")
    building_info = load_building_info(BUILDING_INFO_PATH)
    print("建筑信息读取完成，数据维度：", building_info.shape)

    print("\n开始读取天气数据...")
    weather = load_epw_weather(EPW_PATH)
    print("天气数据读取完成，数据维度：", weather.shape)

    print("\n开始读取 20 栋建筑能耗数据...")
    energy = load_all_building_energy(DATA_DIR)
    print("建筑能耗数据读取完成，数据维度：", energy.shape)

    print("\n开始合并建筑信息...")
    hourly_df = energy.merge(
        building_info,
        on="building_id",
        how="left"
    )

    print("合并建筑信息后，数据维度：", hourly_df.shape)

    print("\n开始合并天气信息...")
    hourly_df = hourly_df.merge(
        weather,
        on="datetime",
        how="left"
    )

    print("合并天气信息后，数据维度：", hourly_df.shape)

    print("\n开始构造时间特征...")
    hourly_df = add_time_features(hourly_df)

    # 单位面积能耗
    # 这里使用 total_floor_area，即 投影面积 × 楼层数
    hourly_df["load_per_area"] = hourly_df["total_load"] / hourly_df["total_floor_area"]

    print("\n开始生成每日 24 小时聚类矩阵...")
    daily_matrix = make_daily_load_matrix(hourly_df)

    # 数据质量检查
    check_data_quality(hourly_df, daily_matrix)

    # 保存结果
    hourly_output_path = OUTPUT_DIR / "clean_hourly_energy.csv"
    daily_output_path = OUTPUT_DIR / "daily_load_matrix.csv"

    hourly_df.to_csv(
        hourly_output_path,
        index=False,
        encoding="utf-8-sig"
    )

    daily_matrix.to_csv(
        daily_output_path,
        index=False,
        encoding="utf-8-sig"
    )

    print("数据预处理完成。")
    print("逐小时清洗数据已保存到：", hourly_output_path)
    print("每日 24 小时矩阵已保存到：", daily_output_path)


if __name__ == "__main__":
    main()