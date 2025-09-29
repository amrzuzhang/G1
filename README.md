# 大气与土壤湿度预测指南

本仓库提供一个基于确定性大气模型与大语言模型（LLM）后处理的 72 小时土壤湿度预测流程。主要组件位于 `src/atmo_model/` 包中，并附带命令行脚本与单元测试，便于快速体验与二次开发。

## 环境准备

- Python 3.11+
- 推荐使用虚拟环境（`python -m venv .venv && source .venv/bin/activate`）
- 安装依赖：本项目核心代码仅使用标准库，若需运行测试请安装 `pytest`
  ```bash
  pip install -U pip
  pip install pytest
  ```

## 项目结构

```
src/atmo_model/
├── __init__.py               # 导出核心数据结构与管线
├── core.py                   # 大气状态、确定性模型与特征提取
├── llm_postprocessing.py     # LLM 提示构建、响应解析
└── pipeline.py               # 三天土壤湿度预测总管线
scripts/
└── predict_soil_moisture.py  # 命令行脚本
```

## Python API 使用方式

1. 构造初始大气状态 `AtmosphericState`
2. 准备逐小时的天气输入（温度、相对湿度、降水量等）
3. 调用 `predict_three_day_soil_moisture`

```python
import datetime

from atmo_model.core import AtmosphericState
from atmo_model.pipeline import predict_three_day_soil_moisture

initial_state = AtmosphericState(
    timestamp=datetime.datetime(2024, 1, 1, 0, 0, 0),
    temperature_c=18.0,
    relative_humidity=60.0,
    precipitation_mm=0.3,
    soil_moisture=0.45,
)

weather_data = [
    {
        "timestamp": "2024-01-01T00:00:00",
        "temperature_c": 18.0,
        "relative_humidity": 60.0,
        "precipitation_mm": 0.1,
    },
    # ... 共 72 条或更多小时级预报
]

forecast = predict_three_day_soil_moisture(initial_state, weather_data)
print(forecast.values)       # 72 个小时级土壤湿度
print(forecast.metadata)     # 包含特征与提示文本
```

若需要自定义 LLM 客户端，可构造 `LLMPostProcessor(client=...)` 并传入 `predict_three_day_soil_moisture` 的 `post_processor` 参数。客户端需接受字符串提示并返回可解析的响应（JSON 字符串/字典/包含数字的文本）。

## 命令行脚本

脚本读取两个 JSON 文件：
- `initial_state.json`：单个初始大气状态
- `weather_data.json`：逐小时天气预报列表（或包含 `{"weather": [...]}` 的对象）

示例 `initial_state.json`：
```json
{
  "timestamp": "2024-01-01T00:00:00",
  "temperature_c": 18.0,
  "relative_humidity": 60.0,
  "precipitation_mm": 0.3,
  "soil_moisture": 0.45
}
```

示例 `weather_data.json`：
```json
[
  {
    "timestamp": "2024-01-01T00:00:00",
    "temperature_c": 18.0,
    "relative_humidity": 60.0,
    "precipitation_mm": 0.1
  },
  {
    "timestamp": "2024-01-01T01:00:00",
    "temperature_c": 18.2,
    "relative_humidity": 59.5,
    "precipitation_mm": 0.0
  }
]
```

运行命令：
```bash
python scripts/predict_soil_moisture.py initial_state.json weather_data.json --output forecast.json
```

脚本会输出包含 72 小时土壤湿度的 `forecast.json` 文件，并在标准输出提示保存路径。

## 运行测试

项目自带 `pytest` 测试，覆盖 LLM 提示生成、响应解析与整体管线。
```bash
pytest
```

## 下一步

- 集成真实的 LLM API：实现一个调用 Hugging Face Transformers 或其他服务的客户端，并注入到 `LLMPostProcessor`。
- 扩展天气要素：在 `load_numerical_weather_outputs` 与 `export_soil_moisture_features` 中添加更多特征。
- 部署：将命令行脚本封装为服务或定时任务，实现持续的土壤湿度预测。
