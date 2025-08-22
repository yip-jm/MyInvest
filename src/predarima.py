import pandas as pd
import numpy as np
import warnings
import os
import glob
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from tqdm import tqdm

# --- 全局设置 ---
warnings.filterwarnings("ignore")

# --- 核心建模与预测函数 (这部分与之前版本相同) ---

def fit_predict_arima_garch(train_data):
    """
    在给定的训练数据上拟合ARIMA-GARCH模型并进行单步预测。
    """
    try:
        p_value = adfuller(train_data)[1]
        d = 1 if p_value >= 0.05 else 0
        arima_model = ARIMA(train_data, order=(1, d, 1))
        arima_fit = arima_model.fit()
        residuals = arima_fit.resid
        garch_model = arch_model(residuals, vol='Garch', p=1, q=1)
        garch_fit = garch_model.fit(disp='off')
        forecast_mean = arima_fit.forecast(steps=1).iloc[0]
        garch_forecast = garch_fit.forecast(horizon=1)
        forecast_variance = garch_forecast.variance.iloc[-1, 0]
        return {"mean": forecast_mean, "variance": forecast_variance}
    except Exception:
        return None

def run_backtest(full_series, window_size, min_backtest_points=20):
    """
    执行滚动窗口回测来计算方向预测的准确率。
    """
    if len(full_series) < window_size + min_backtest_points:
        return np.nan

    correct_predictions = 0
    total_predictions = 0

    for i in range(window_size, len(full_series)):
        train_data = full_series.iloc[i - window_size : i]
        actual_value = full_series.iloc[i]
        try:
            p_value = adfuller(train_data)[1]
            d = 1 if p_value >= 0.05 else 0
            arima_model = ARIMA(train_data, order=(1, d, 1))
            arima_fit = arima_model.fit()
            predicted_mean = arima_fit.forecast(steps=1).iloc[0]
            if np.sign(predicted_mean) == np.sign(actual_value):
                correct_predictions += 1
            total_predictions += 1
        except Exception:
            continue
    
    return correct_predictions / total_predictions if total_predictions > 0 else np.nan

def process_fund_file(file_path, window_sizes):
    """
    处理单个基金文件：进行回测和最终预测。
    """
    try:
        fund_code = os.path.basename(file_path).split('.')[0]
        df = pd.read_csv(file_path, parse_dates=['FSRQ'])
        df.dropna(subset=['JZZZL', 'DWJZ'], inplace=True)
        df = df.sort_values(by='FSRQ').reset_index(drop=True)

        if df.empty:
            print(f"警告: 文件 {fund_code} 为空或无有效数据，已跳过。")
            return []

        returns = df.set_index('FSRQ')['JZZZL']
        last_date = df['FSRQ'].iloc[-1]
        last_dwjz = df['DWJZ'].iloc[-1]
        prediction_date = last_date + pd.Timedelta(days=1)
        
        fund_results = []
        for window in window_sizes:
            if len(returns) < window:
                result = {
                    "fund_code": fund_code, "window_size": window,
                    "error_message": f"数据不足 (仅 {len(returns)} 条)",
                }
                fund_results.append(result)
                continue

            accuracy = run_backtest(returns, window)
            prediction = fit_predict_arima_garch(returns.iloc[-window:])
            
            if prediction:
                result = {
                    "fund_code": fund_code, "window_size": window,
                    "last_known_date": last_date.strftime('%Y-%m-%d'),
                    "last_known_dwjz": last_dwjz,
                    "prediction_date": prediction_date.strftime('%Y-%m-%d'),
                    "predicted_jzzzl": prediction["mean"],
                    "predicted_dwjz": last_dwjz * (1 + prediction["mean"] / 100),
                    "predicted_volatility_std": np.sqrt(prediction["variance"]),
                    "directional_accuracy": accuracy, "error_message": None,
                }
            else:
                result = {
                    "fund_code": fund_code, "window_size": window,
                    "error_message": "最终预测模型拟合失败",
                }
            fund_results.append(result)
        
        return fund_results
    except Exception as e:
        fund_code = os.path.basename(file_path).split('.')[0]
        print(f"处理文件 {fund_code} 时发生严重错误: {e}")
        return [{"fund_code": fund_code, "error_message": str(e)}]

# --- 主处理流程 (这部分已更新) ---

def main():
    """
    主执行函数：扫描文件，为每个文件独立处理并保存结果。
    """
    # --- 用户可配置参数 ---
    DATA_DIRECTORY = 'data-china'  # 存放基金CSV文件的子文件夹
    PREDICTION_DIRECTORY = 'prediction'  # 存放所有独立结果文件的文件夹
    WINDOW_SIZES_TO_TEST = [30, 90, 180]  # 要测试的窗口大小

    # 1. 确保输出目录存在
    os.makedirs(PREDICTION_DIRECTORY, exist_ok=True)

    # 2. 查找所有输入CSV文件
    csv_files = glob.glob(os.path.join(DATA_DIRECTORY, '*.csv'))
    
    if not csv_files:
        print(f"错误：在 '{DATA_DIRECTORY}' 文件夹中没有找到任何 .csv 文件。")
        return

    # 3. 循环处理每个文件，并立即保存结果
    for file_path in tqdm(csv_files, desc="处理所有基金文件"):
        # (A) 对当前文件进行预测和回测
        fund_results = process_fund_file(file_path, WINDOW_SIZES_TO_TEST)

        # (B) 如果有结果，则将其保存到对应的文件中
        if fund_results:
            results_df = pd.DataFrame(fund_results)
            
            # (C) 确定输出文件的路径和名称
            base_filename = os.path.basename(file_path)
            output_path = os.path.join(PREDICTION_DIRECTORY, base_filename)
            
            # (D) 整理列顺序并保存
            cols_order = [
                "fund_code", "window_size", "prediction_date", "predicted_dwjz",
                "predicted_jzzzl", "directional_accuracy", "predicted_volatility_std",
                "last_known_date", "last_known_dwjz", "error_message"
            ]
            for col in cols_order:
                if col not in results_df.columns:
                    results_df[col] = np.nan
            results_df = results_df[cols_order]

            results_df.to_csv(output_path, index=False, float_format='%.4f')
    
    # 4. 打印最终的总结信息
    line_length = 60
    print("\n" + "=" * line_length)
    print(" " * 22 + "处理全部完成！")
    print(f"所有基金的独立预测结果文件均已保存到 '{PREDICTION_DIRECTORY}' 文件夹中。")
    print("=" * line_length)


if __name__ == '__main__':
    main()