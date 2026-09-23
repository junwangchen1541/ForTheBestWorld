# 沪深300周度轮动

一个面向研究和模拟交易的量化项目：三种模型分别在全 A 股和沪深300两个训练股票池拟合，形成 3×2 六组实验；每组都以独立 20 万元账户，在共同的沪深300样本外区间进行周度轮动比较。

比较面板公开六组的净值、收益、回撤、训练信息、持仓和全部成交。模型为梯度提升树、Laya 决策模型加股票池专属校准器、前馈神经网络。

> 这不是投资建议，也不保证收益。默认下载器使用“当前”沪深300成分股，历史回测会有幸存者偏差；用于实盘决策前，应接入历史时点成分股、停复牌、涨跌停和公司行动数据。

## 策略约定

- 标签：未来 20 个交易日收益；输入特征包括 5/20/60/120 日动量、波动率、回撤、成交额和日内振幅。
- 训练池：全 A 股与沪深300；训练截止日、预测周期和最小训练样本数见 `[model]` 配置。
- 评估：六组统一只在沪深300成分范围交易，保证收益可横向比较；必须确保评估行情覆盖训练截止日前后的特征历史。
- Laya：使用官方多语言预训练检查点冻结推理，再按训练池拟合收益校准器；不是重新训练 Laya 权重。当前配置用 2023-12-29 作为训练截止日、2024 年起做样本外测试；这段历史数据用于评估策略机制，不把 Laya 发布日误当成历史数据的起点。
- 调仓：每周最后一个交易日收盘后评分，下一交易日开盘执行。
- 持仓：10 只；每周默认至少替换 1 只、最多 2 只。
- 交易：A 股 100 股整手，买卖佣金最低 5 元，卖出收印花税。
- 仓位：首次建仓和新标的按组合净值的 1/10 分配；未成交资金保留为现金。

## 安装

Windows PowerShell：

```powershell
py -m venv .venv
.venv\Scripts\python -m pip install -e ".[data,dev]"
Copy-Item config.example.toml config.toml
```

下载当前沪深300成分股的前复权日线（使用 AkShare 的腾讯历史行情接口，支持断点续传）：

```powershell
.venv\Scripts\hs300-rotation download --start 20180101 --end 20260923
```

运行回测：

```powershell
.venv\Scripts\hs300-rotation backtest --config config.toml --data data/hs300.csv
```

结果写入 `output/equity.csv`、`output/trades.csv` 和 `output/metrics.json`。

分别下载当前全 A 股训练数据和沪深300行情：

```powershell
.venv\Scripts\hs300-rotation download --universe all-a --start 20180101 --end 20260923 --output data/all_a.csv
.venv\Scripts\hs300-rotation download --universe hs300 --start 20180101 --end 20260923 --output data/hs300.csv
```

安装 Laya 可选运行依赖（首次加载需下载约数 GB 权重）：

```powershell
.venv\Scripts\python -m pip install -e ".[laya]"
```

训练并运行六组模型实验：

```powershell
.venv\Scripts\hs300-rotation league --config config.toml --training-data data/all_a.csv --evaluation-data data/hs300.csv
```

测试会从 `[model].training_end_date` 之后开始，以免把训练标签区间作为样本外收益。六组数据输出到 `dashboard/data.js`；面板初次打开显示明确标记的示例实验。下载器使用当前成分股，因此历史测试仍有幸存者偏差；但行情本身来自真实历史数据，不是合成数据。

生成最新评分和调仓建议：

```powershell
.venv\Scripts\hs300-rotation recommend --config config.toml --data data/hs300.csv --holdings 600519,000858,601318
```

建议输出只表达目标股票变化，不会连接券商或自动下单。排名明细写入 `output/latest_ranking.csv`。

## 自备数据

CSV 使用长表格式，每行一只股票的一个交易日：

```text
date,symbol,open,close,volume,amount
2025-01-02,600519,1515.00,1521.50,2389100,3621000000
```

价格必须采用一致的复权口径。当前下载器拉取当前成分股，不能提供历史时点全A股票池，历史样本会有幸存者偏差；严谨实验应使用点时成分、退市样本、停复牌和涨跌停成交约束。全A股数据量很大，完整下载需要较长时间。

## 测试

```powershell
.venv\Scripts\python -m pytest
.venv\Scripts\python -m ruff check .
```
