import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os
import glob
import warnings

# 忽略一些常见的警告，使输出更整洁
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def create_features(df, window_size):
    """
    根据给定的DataFrame和窗口大小创建特征。
    """
    df_featured = df.copy()

    # 1. 创建日期特征
    df_featured['day_of_week'] = df_featured['FSRQ'].dt.dayofweek
    df_featured['day_of_year'] = df_featured['FSRQ'].dt.dayofyear
    df_featured['month'] = df_featured['FSRQ'].dt.month
    df_featured['week_of_year'] = df_featured['FSRQ'].dt.isocalendar().week.astype(int)

    # 2. 创建滞后和滑动窗口特征
    for i in range(1, window_size + 1):
        df_featured[f'DWJZ_lag_{i}'] = df_featured['DWJZ'].shift(i)
        df_featured[f'JZZZL_lag_{i}'] = df_featured['JZZZL'].shift(i)

    df_featured[f'DWJZ_roll_mean'] = df_featured['DWJZ'].shift(1).rolling(window=window_size).mean()
    df_featured[f'DWJZ_roll_std'] = df_featured['DWJZ'].shift(1).rolling(window=window_size).std()
    df_featured[f'JZZZL_roll_mean'] = df_featured['JZZZL'].shift(1).rolling(window=window_size).mean()
    
    # 使用向后填充来处理因滑动窗口在数据开头产生的NaN
    # 这比前向填充更合理，因为它使用了未来的真实数据来填充历史的统计值
    df_featured.fillna(method='bfill', inplace=True)
    
    return df_featured.dropna().reset_index(drop=True)

def calculate_direction_accuracy(y_true, y_pred, X_test):
    """计算方向预测准确率"""
    # 对于DWJZ，方向是与前一天比较
    actual_direction_dwjz = np.sign(y_true['DWJZ'] - X_test['DWJZ_lag_1'])
    predicted_direction_dwjz = np.sign(y_pred['DWJZ'] - X_test['DWJZ_lag_1'])
    
    # 对于JZZZL，方向是其本身的正负
    actual_direction_jzzzl = np.sign(y_true['JZZZL'])
    predicted_direction_jzzzl = np.sign(y_pred['JZZZL'])

    # 将0（表示不变）替换为1（表示上涨），以避免在比较中产生偏差
    actual_direction_dwjz[actual_direction_dwjz == 0] = 1
    predicted_direction_dwjz[predicted_direction_dwjz == 0] = 1
    actual_direction_jzzzl[actual_direction_jzzzl == 0] = 1
    predicted_direction_jzzzl[predicted_direction_jzzzl == 0] = 1

    acc_dwjz = np.mean(actual_direction_dwjz.values == predicted_direction_dwjz.values)
    acc_jzzzl = np.mean(actual_direction_jzzzl.values == predicted_direction_jzzzl.values)
    
    return acc_dwjz, acc_jzzzl


def process_fund_file(file_path, window_sizes):
    """
    处理单个基金文件，返回包含所有窗口结果的DataFrame。
    """
    fund_code = os.path.basename(file_path).split('.')[0]
    print(f"--- 正在处理基金: {fund_code} ---")
    
    try:
        # 尝试使用不同的编码读取，以增加兼容性
        df = pd.read_csv(file_path, encoding='utf-8')
    except Exception:
        try:
            df = pd.read_csv(file_path, encoding='gbk')
        except Exception as e:
            print(f"错误：无法读取文件 {file_path}: {e}")
            return None

    # 1. 基础数据清洗
    df_processed = df[['FSRQ', 'DWJZ', 'JZZZL', 'SGZT', 'SHZT']].copy()
    df_processed['FSRQ'] = pd.to_datetime(df_processed['FSRQ'])
    df_processed = df_processed.sort_values('FSRQ').reset_index(drop=True)
    df_processed.dropna(subset=['DWJZ', 'JZZZL'], inplace=True)

    # 对分类变量进行编码
    sgzt_encoder = LabelEncoder().fit(df_processed['SGZT'])
    shzt_encoder = LabelEncoder().fit(df_processed['SHZT'])
    df_processed['SGZT_encoded'] = sgzt_encoder.transform(df_processed['SGZT'])
    df_processed['SHZT_encoded'] = shzt_encoder.transform(df_processed['SHZT'])

    all_results = []

    # 2. 循环处理每个窗口
    for window in window_sizes:
        print(f"  使用窗口: {window} 天...")
        # 数据量必须足够大以创建特征和测试集
        if len(df_processed) < window + 15: # 至少需要约15条数据用于测试
            print(f"  数据量不足 ({len(df_processed)}行)，跳过 {window}天 窗口。")
            continue

        # 3. 特征工程
        df_featured = create_features(df_processed, window)
        
        if df_featured.empty:
            print(f"  特征工程后数据为空，跳过 {window}天 窗口。")
            continue

        features = [col for col in df_featured.columns if col not in ['FSRQ', 'DWJZ', 'JZZZL', 'SGZT', 'SHZT']]
        targets = ['DWJZ', 'JZZZL']
        
        X = df_featured[features]
        y = df_featured[targets]

        # 4. 评估准确率 (使用最后20%的数据作为测试集)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        if len(X_test) == 0:
            print(f"  分割后测试集为空，跳过 {window}天 窗口评估。")
            continue
        
        # 训练模型
        model_dwjz_eval = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
        model_dwjz_eval.fit(X_train, y_train['DWJZ'])
        
        model_jzzzl_eval = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
        model_jzzzl_eval.fit(X_train, y_train['JZZZL'])
        
        # 预测测试集
        y_pred_dwjz = model_dwjz_eval.predict(X_test)
        y_pred_jzzzl = model_jzzzl_eval.predict(X_test)
        y_pred_df = pd.DataFrame({'DWJZ': y_pred_dwjz, 'JZZZL': y_pred_jzzzl}, index=y_test.index)

        # 计算方向准确率
        accuracy_dwjz, accuracy_jzzzl = calculate_direction_accuracy(y_test, y_pred_df, X_test)
        
        # 5. 最终预测 (使用全部数据重新训练)
        model_dwjz_final = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
        model_dwjz_final.fit(X, y['DWJZ'])
        
        model_jzzzl_final = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
        model_jzzzl_final.fit(X, y['JZZZL'])

        # 构建下一天的特征向量
        last_known_date = df_processed['FSRQ'].max()
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

        # 执行预测
        predicted_dwjz = model_dwjz_final.predict(future_df)[0]
        predicted_jzzzl = model_jzzzl_final.predict(future_df)[0]

        # 6. 存储结果
        all_results.append({
            'window_size': window,
            'direction_accuracy_dwjz': f"{accuracy_dwjz:.2%}",
            'direction_accuracy_jzzzl': f"{accuracy_jzzzl:.2%}",
            'next_day_prediction_date': prediction_date.strftime('%Y-%m-%d'),
            'predicted_dwjz': f"{predicted_dwjz:.4f}",
            'predicted_jzzzl_pct': f"{predicted_jzzzl:.2f}"
        })

    return pd.DataFrame(all_results)


def main():
    """
    主函数，执行所有操作。
    """
    # 定义路径和参数
    INPUT_DIR = '../data-china/funds'
    OUTPUT_DIR = 'prediction_xgboost'
    WINDOW_SIZES = [7, 30, 90, 180]

    # 检查输入目录是否存在，如果不存在则创建并退出
    if not os.path.exists(INPUT_DIR):
        os.makedirs(INPUT_DIR)
        print(f"输入文件夹 '{INPUT_DIR}' 不存在，已为您创建。")
        print("请将您的基金CSV文件放入此文件夹中，然后重新运行脚本。")
        return

    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 获取所有CSV文件
    fund_files = glob.glob(os.path.join(INPUT_DIR, '*.csv'))

    if not fund_files:
        print(f"在文件夹 '{INPUT_DIR}' 中没有找到任何 .csv 文件。请添加文件后再运行。")
        return

    # 循环处理每个文件
    for file_path in fund_files:
        results_df = process_fund_file(file_path, WINDOW_SIZES)
        
        if results_df is not None and not results_df.empty:
            fund_code = os.path.basename(file_path)
            output_path = os.path.join(OUTPUT_DIR, fund_code)
            results_df.to_csv(output_path, index=False)
            print(f"结果已保存至: {output_path}\n")
        else:
            print(f"基金 {os.path.basename(file_path)} 未生成任何结果（可能由于数据量不足）。\n")

    print("所有基金处理完毕！")

if __name__ == '__main__':
    main()