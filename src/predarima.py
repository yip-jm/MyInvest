# -*- coding: utf-8 -*-
# ARIMA-GARCH 滑动窗口滚动预测 DWJZ 和 JZZZL（%）
# 输出 RMSE/MAE/MAPE/方向命中率
# 依赖: pandas, numpy, pmdarima, arch, sklearn

import pandas as pd
import numpy as np
from pmdarima import auto_arima
from arch import arch_model
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---------------------------
# 数据处理函数
# ---------------------------
def load_series(csv_path: str, date_col: str = "FSRQ", price_col: str = "DWJZ") -> pd.Series:
    df = pd.read_csv(csv_path)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col, price_col])
    df = df.sort_values(date_col)
    s = pd.to_numeric(df[price_col], errors="coerce").dropna()
    s.index = df.loc[s.index, date_col].values
    s = s[s > 0]
    if len(s) < 20:
        print("⚠️ 样本数较少，预测可能不稳定。")
    return s

def to_log_returns(price: pd.Series) -> pd.Series:
    return np.log(price).diff().dropna()

# ---------------------------
# 模型拟合函数
# ---------------------------
def fit_arima_mean(r: pd.Series):
    arima = auto_arima(
        r, seasonal=False, stepwise=True,
        suppress_warnings=True, error_action="ignore",
        information_criterion="aic", max_p=3, max_q=3, max_d=1
    )
    mu_forecast, conf_int = arima.predict(n_periods=1, return_conf_int=True, alpha=0.05)
    mu_hat = float(mu_forecast[0])
    mu_in_sample = arima.predict_in_sample()
    resid = r.iloc[-len(mu_in_sample):] - pd.Series(mu_in_sample, index=r.index[-len(mu_in_sample):])
    return arima, mu_hat, resid

def fit_garch_variance(resid: pd.Series):
    scale = 100.0
    am = arch_model(resid * scale, mean="Zero", vol="GARCH", p=1, q=1, dist="normal")
    res = am.fit(disp="off")
    fcast = res.forecast(horizon=1)
    var_next_scaled = fcast.variance.values[-1, 0]
    var_next = (var_next_scaled / (scale ** 2))
    sigma_next = float(np.sqrt(var_next))
    return res, sigma_next

def combine_to_price_and_pct(last_price: float, mu_hat: float, sigma_hat: float, z: float = 1.96):
    next_price = last_price * np.exp(mu_hat)
    next_pct = (np.exp(mu_hat) - 1.0) * 100.0
    lo_r = mu_hat - z * sigma_hat
    hi_r = mu_hat + z * sigma_hat
    price_lo = last_price * np.exp(lo_r)
    price_hi = last_price * np.exp(hi_r)
    pct_lo = (np.exp(lo_r) - 1.0) * 100.0
    pct_hi = (np.exp(hi_r) - 1.0) * 100.0
    return {
        "pred_price": float(next_price),
        "price_ci95": (float(price_lo), float(price_hi)),
        "pred_pct": float(next_pct),
        "pct_ci95": (float(pct_lo), float(pct_hi))
    }

# ---------------------------
# 滑动窗口滚动预测
# ---------------------------
def rolling_arima_garch(csv_path: str, window_size: int = 200, date_col: str = "FSRQ", price_col: str = "DWJZ"):
    price = load_series(csv_path, date_col, price_col)
    true_prices = []
    pred_prices = []
    pred_pct = []
    true_pct = []
    last_price = None

    for i in range(window_size, len(price)):
        train_window = price.iloc[i-window_size:i]
        r_window = to_log_returns(train_window)
        if len(r_window) < 5:  # 防止数据太少
            continue

        # ARIMA 均值
        arima_model, mu_hat, resid = fit_arima_mean(r_window)
        # GARCH 方差
        garch_model, sigma_hat = fit_garch_variance(resid)

        # 预测下一日
        last_price = train_window.iloc[-1]
        out = combine_to_price_and_pct(last_price, mu_hat, sigma_hat)

        # 保存
        true_price = price.iloc[i]
        true_prices.append(true_price)
        pred_prices.append(out["pred_price"])

        true_r = np.log(true_price / last_price)
        true_pct.append(true_r * 100)
        pred_pct.append(out["pred_pct"])

    # ---------------------------
    # 回测指标
    # ---------------------------
    true_prices = np.array(true_prices)
    pred_prices = np.array(pred_prices)
    true_pct = np.array(true_pct)
    pred_pct = np.array(pred_pct)

    mae_price = mean_absolute_error(true_prices, pred_prices)
    rmse_price = np.sqrt(mean_squared_error(true_prices, pred_prices))
    mape_price = np.mean(np.abs((true_prices - pred_prices) / true_prices)) * 100

    mae_pct = mean_absolute_error(true_pct, pred_pct)
    rmse_pct = np.sqrt(mean_squared_error(true_pct, pred_pct))
    mape_pct = np.mean(np.abs((true_pct - pred_pct) / (np.abs(true_pct)+1e-8))) * 100

    # 涨跌方向命中率
    direction_true = np.sign(true_pct)
    direction_pred = np.sign(pred_pct)
    direction_acc = (direction_true == direction_pred).mean() * 100

    metrics = {
        "RMSE_price": rmse_price,
        "MAE_price": mae_price,
        "MAPE_price%": mape_price,
        "RMSE_pct": rmse_pct,
        "MAE_pct": mae_pct,
        "MAPE_pct%": mape_pct,
        "Direction_acc%": direction_acc,
        "num_predictions": len(pred_prices)
    }

    return pred_prices, pred_pct, metrics

# ---------------------------
# 主函数示例
# ---------------------------
if __name__ == "__main__":
    csv_path = "data-china\\004246.csv"  # 请替换为你的完整历史数据文件
    window_size = 90      # 可调整窗口大小
    pred_prices, pred_pct, metrics = rolling_arima_garch(csv_path, window_size)

    print("\n=== 滑动窗口滚动预测指标 ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    # 可选：保存预测序列
    df_pred = pd.DataFrame({
        "pred_DWJZ": pred_prices,
        "pred_JZZZL%": pred_pct
    })
    df_pred.to_csv("rolling_pred.csv", index=False)
    print("\n预测序列已保存到 rolling_pred.csv")
