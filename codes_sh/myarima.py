import pandas as pd
import numpy as np
import warnings
import os
import sys  # 导入sys模块以读取命令行参数
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model

# --- 全局设置 ---
warnings.filterwarnings("ignore")

# --- 核心建模函数 (与之前完全相同) ---
def fit_predict_arima_garch(train_data):
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
    if len(full_series) < window_size + min_backtest_points:
        return np.nan
    correct_predictions, total_predictions = 0, 0
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

# --- 主处理流程 (修改后) ---
def process_fund_file(file_path, window_sizes, output_dir):
    try:
        fund_code = os.path.basename(file_path).split('.')[0]
        df = pd.read_csv(file_path, parse_dates=['FSRQ'])
        df.dropna(subset=['JZZZL', 'DWJZ'], inplace=True)
        df = df.sort_values(by='FSRQ').reset_index(drop=True)
        if df.empty: return

        returns = df.set_index('FSRQ')['JZZZL']
        last_date = df['FSRQ'].iloc[-1]
        last_dwjz = df['DWJZ'].iloc[-1]
        prediction_date = last_date + pd.Timedelta(days=1)
        
        fund_results = []
        for window in window_sizes:
            # ... (这部分逻辑和之前完全一样) ...
            if len(returns) < window:
                result = {"fund_code": fund_code, "window_size": window, "error_message": f"数据不足"}
            else:
                accuracy = run_backtest(returns, window)
                prediction = fit_predict_arima_garch(returns.iloc[-window:])
                if prediction:
                    result = {
                        "fund_code": fund_code, "window_size": window,
                        "last_known_date": last_date.strftime('%Y-%m-%d'), "last_known_dwjz": last_dwjz,
                        "prediction_date": prediction_date.strftime('%Y-%m-%d'),
                        "predicted_jzzzl": prediction["mean"],
                        "predicted_dwjz": last_dwjz * (1 + prediction["mean"] / 100),
                        "predicted_volatility_std": np.sqrt(prediction["variance"]),
                        "directional_accuracy": accuracy, "error_message": None,
                    }
                else:
                    result = {"fund_code": fund_code, "window_size": window, "error_message": "模型拟合失败"}
            fund_results.append(result)

        if fund_results:
            results_df = pd.DataFrame(fund_results)
            # ... (整理列顺序的逻辑也和之前一样) ...
            cols_order = ["fund_code", "window_size", "prediction_date", "predicted_dwjz", "predicted_jzzzl", "directional_accuracy", "predicted_volatility_std", "last_known_date", "last_known_dwjz", "error_message"]
            for col in cols_order:
                if col not in results_df.columns: results_df[col] = np.nan
            results_df = results_df[cols_order]

            # 将结果保存到指定输出目录
            output_path = os.path.join(output_dir, os.path.basename(file_path))
            results_df.to_csv(output_path, index=False, float_format='%.4f')

    except Exception as e:
        print(f"处理文件 {file_path} 时发生严重错误: {e}")

def main():
    # 检查命令行参数是否足够
    if len(sys.argv) != 2:
        print("用法: python process_single_fund.py <path_to_input_csv>")
        sys.exit(1)

    # 从命令行获取输入文件路径
    input_file_path = sys.argv[1]
    
    # 定义固定参数
    output_directory = 'predictions_arima'
    window_sizes_to_test = [30, 90, 180]

    # 确保输出目录存在
    os.makedirs(output_directory, exist_ok=True)

    # 调用处理函数
    process_fund_file(input_file_path, window_sizes_to_test, output_directory)

if __name__ == '__main__':
    main()