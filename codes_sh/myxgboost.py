import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os
import glob
import warnings
import argparse  # <-- 1. 导入 argparse 模块


def create_features(df, window_size):
    """
    根据给定的DataFrame和窗口大小创建特征 (优化后版本)。
    """
    df_featured = df.copy()

    # 1. 创建一个字典来收集所有新的特征列
    features_to_add = {}

    # 2. 生成日期特征并放入字典
    features_to_add['day_of_week'] = df_featured['FSRQ'].dt.dayofweek
    features_to_add['day_of_year'] = df_featured['FSRQ'].dt.dayofyear
    features_to_add['month'] = df_featured['FSRQ'].dt.month
    features_to_add['week_of_year'] = df_featured['FSRQ'].dt.isocalendar().week.astype(int)

    # 3. 在循环中生成滞后特征并放入字典
    for i in range(1, window_size + 1):
        features_to_add[f'DWJZ_lag_{i}'] = df_featured['DWJZ'].shift(i)
        features_to_add[f'JZZZL_lag_{i}'] = df_featured['JZZZL'].shift(i)

    # 4. 生成滑动窗口特征并放入字典
    features_to_add[f'DWJZ_roll_mean'] = df_featured['DWJZ'].shift(1).rolling(window=window_size).mean()
    features_to_add[f'DWJZ_roll_std'] = df_featured['DWJZ'].shift(1).rolling(window=window_size).std()
    features_to_add[f'JZZZL_roll_mean'] = df_featured['JZZZL'].shift(1).rolling(window=window_size).mean()
    
    # 5. 将包含所有新特征的字典转换为DataFrame
    new_features_df = pd.DataFrame(features_to_add)
    
    # 6. 使用 pd.concat 一次性将所有新特征合并到原始DataFrame
    df_final = pd.concat([df_featured, new_features_df], axis=1)

    # 7. 填充并清理
    # (之前的 .fillna/.bfill 逻辑)
    df_final = df_final.bfill()
    
    return df_final.dropna().reset_index(drop=True)

def calculate_direction_accuracy(y_true, y_pred, X_test):
    """计算方向预测准确率"""
    actual_direction_dwjz = np.sign(y_true['DWJZ'] - X_test['DWJZ_lag_1'])
    predicted_direction_dwjz = np.sign(y_pred['DWJZ'] - X_test['DWJZ_lag_1'])
    actual_direction_jzzzl = np.sign(y_true['JZZZL'])
    predicted_direction_jzzzl = np.sign(y_pred['JZZZL'])
    actual_direction_dwjz[actual_direction_dwjz == 0] = 1
    predicted_direction_dwjz[predicted_direction_dwjz == 0] = 1
    actual_direction_jzzzl[actual_direction_jzzzl == 0] = 1
    predicted_direction_jzzzl[predicted_direction_jzzzl == 0] = 1
    acc_dwjz = np.mean(actual_direction_dwjz.values == predicted_direction_dwjz.values)
    acc_jzzzl = np.mean(actual_direction_jzzzl.values == predicted_direction_jzzzl.values)
    return acc_dwjz, acc_jzzzl

def process_fund_file(file_path, window_sizes, output_dir):
    """
    处理单个基金文件，返回包含所有窗口结果的DataFrame。
    """
    fund_code = os.path.basename(file_path).split('.')[0]
    print(f"--- 正在处理基金: {fund_code} ---")
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except Exception:
        try:
            df = pd.read_csv(file_path, encoding='gbk')
        except Exception as e:
            print(f"错误：无法读取文件 {file_path}: {e}")
            return
    df_processed = df[['FSRQ', 'DWJZ', 'JZZZL', 'SGZT', 'SHZT']].copy()
    df_processed['FSRQ'] = pd.to_datetime(df_processed['FSRQ'])
    df_processed = df_processed.sort_values('FSRQ').reset_index(drop=True)
    df_processed.dropna(subset=['DWJZ', 'JZZZL'], inplace=True)
    sgzt_encoder = LabelEncoder().fit(df_processed['SGZT'])
    shzt_encoder = LabelEncoder().fit(df_processed['SHZT'])
    df_processed['SGZT_encoded'] = sgzt_encoder.transform(df_processed['SGZT'])
    df_processed['SHZT_encoded'] = shzt_encoder.transform(df_processed['SHZT'])
    all_results = []
    for window in window_sizes:
        print(f"  使用窗口: {window} 天...")
        if len(df_processed) < window + 15:
            print(f"  数据量不足 ({len(df_processed)}行)，跳过 {window}天 窗口。")
            continue
        df_featured = create_features(df_processed, window)
        if df_featured.empty:
            print(f"  特征工程后数据为空，跳过 {window}天 窗口。")
            continue
        features = [col for col in df_featured.columns if col not in ['FSRQ', 'DWJZ', 'JZZZL', 'SGZT', 'SHZT']]
        targets = ['DWJZ', 'JZZZL']
        X = df_featured[features]
        y = df_featured[targets]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        if len(X_test) == 0:
            print(f"  分割后测试集为空，跳过 {window}天 窗口评估。")
            continue
        model_dwjz_eval = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
        model_dwjz_eval.fit(X_train, y_train['DWJZ'])
        model_jzzzl_eval = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
        model_jzzzl_eval.fit(X_train, y_train['JZZZL'])
        y_pred_dwjz = model_dwjz_eval.predict(X_test)
        y_pred_jzzzl = model_jzzzl_eval.predict(X_test)
        y_pred_df = pd.DataFrame({'DWJZ': y_pred_dwjz, 'JZZZL': y_pred_jzzzl}, index=y_test.index)
        accuracy_dwjz, accuracy_jzzzl = calculate_direction_accuracy(y_test, y_pred_df, X_test)
        model_dwjz_final = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
        model_dwjz_final.fit(X, y['DWJZ'])
        model_jzzzl_final = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
        model_jzzzl_final.fit(X, y['JZZZL'])
        last_known_date = pd.to_datetime(df_processed['FSRQ'].max())
        prediction_date = last_known_date + pd.Timedelta(days=1)
        recent_data_for_features = df_processed.tail(window)
        future_features = {}
        future_features['day_of_week'] = prediction_date.dayofweek
        future_features['day_of_year'] = prediction_date.dayofyear
        future_features['month'] = prediction_date.month
        future_features['week_of_year'] = prediction_date.isocalendar().week
        for i in range(1, window + 1):
            future_features[f'DWJZ_lag_{i}'] = df_processed['DWJZ'].iloc[-i]
            future_features[f'JZZZL_lag_{i}'] = df_processed['JZZZL'].iloc[-i]
        future_features[f'DWJZ_roll_mean'] = recent_data_for_features['DWJZ'].mean()
        future_features[f'DWJZ_roll_std'] = recent_data_for_features['DWJZ'].std()
        future_features[f'JZZZL_roll_mean'] = recent_data_for_features['JZZZL'].mean()
        future_features['SGZT_encoded'] = df_processed['SGZT_encoded'].iloc[-1]
        future_features['SHZT_encoded'] = df_processed['SHZT_encoded'].iloc[-1]
        future_df = pd.DataFrame([future_features])[X.columns]
        predicted_dwjz = model_dwjz_final.predict(future_df)[0]
        predicted_jzzzl = model_jzzzl_final.predict(future_df)[0]
        all_results.append({
            'window_size': window,
            'next_day_prediction_date': prediction_date.strftime('%Y-%m-%d'),
            'predicted_dwjz': f"{predicted_dwjz:.4f}",
            'predicted_jzzzl_pct': f"{predicted_jzzzl:.2f}",
            'direction_accuracy_dwjz': f"{accuracy_dwjz:.2%}",
            'direction_accuracy_jzzzl': f"{accuracy_jzzzl:.2%}"     
        })
    results_df = pd.DataFrame(all_results)
    if not results_df.empty:
        output_filename = os.path.basename(file_path)
        output_path = os.path.join(output_dir, output_filename)
        results_df.to_csv(output_path, index=False)
        print(f"结果已保存至: {output_path}\n")
    else:
        print(f"基金 {fund_code} 未生成任何结果。\n")


def main():
    """
    主函数，执行所有操作。
    """
    # --- 2. 设置命令行参数解析 ---
    parser = argparse.ArgumentParser(description="对给定的基金CSV文件列表进行下一日净值预测。")
    parser.add_argument('files', nargs='+', help='需要处理的一个或多个基金CSV文件的路径。')
    args = parser.parse_args()
    
    # 定义路径和参数
    OUTPUT_DIR = 'prediction_xgboost'
    WINDOW_SIZES = [7, 30, 90, 180]

    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- 3. 使用从命令行传入的文件列表 ---
    fund_files = args.files

    if not fund_files:
        print(f"错误：没有提供任何文件进行处理。")
        return

    # 循环处理每个文件
    for file_path in fund_files:
        process_fund_file(file_path, WINDOW_SIZES, OUTPUT_DIR)

    print("此批次的所有文件处理完毕！")

if __name__ == '__main__':
    main()