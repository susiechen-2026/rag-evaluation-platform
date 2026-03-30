# RAG评测平台

RAG（Retrieval-Augmented Generation）评测平台是一个用于评估和优化RAG系统性能的工具。它可以帮助你评测RAG系统在多个维度上的表现，并提供详细的优化建议。

## 功能特性

- **多维度评测**：评估RAG系统的检索相关性、生成质量、事实一致性等多个维度
- **性能指标**：测量系统的响应时间、Token效率和成本
- **并行处理**：支持并行评测，提高评测速度
- **缓存机制**：避免重复计算，提高系统性能
- **可视化分析**：生成评测结果的可视化图表
- **详细优化建议**：基于评测结果提供针对性的优化建议
- **配置管理**：通过配置文件管理所有配置项

## 安装

1. 克隆项目到本地：

```bash
git clone <repository-url>
cd rag-evaluation-platform
```

2. 创建并激活虚拟环境：

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. 安装依赖：

```bash
pip install -r requirements.txt
```

## 配置

1. 复制配置文件模板并修改：

```bash
cp config.json config.json
```

2. 编辑 `config.json` 文件，设置你的API密钥和其他配置：

```json
{
  "api": {
    "dashscope_api_key": "YOUR_API_KEY_HERE"
  },
  "model": {
    "name": "qwen-plus",
    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "temperature": 0
  },
  "evaluation": {
    "threshold": 0.7,
    "batch_size": 5,
    "max_workers": 4
  },
  "output": {
    "eval_results_dir": "./eval_results",
    "visualizations_dir": "./visualizations"
  },
  "rag": {
    "type": "local",
    "local_function": "rag.rag_observe.rag_chatbot"
  }
}
```

或者，你也可以通过环境变量设置API密钥：

```bash
# Windows
set DASHSCOPE_API_KEY=your-api-key
# Linux/Mac
export DASHSCOPE_API_KEY=your-api-key
```

## 使用方法

1. 准备测试数据集：

   测试数据集应该是一个JSON文件，格式如下：

   ```json
   {
     "goldens": [
       {
         "input": "问题1",
         "expected_output": "期望答案1"
       },
       {
         "input": "问题2",
         "expected_output": "期望答案2"
       }
     ]
   }
   ```

   默认情况下，系统会使用 `data/dataset.json` 文件作为测试数据集。

2. 运行评测：

```bash
python main.py
```

3. 查看评测结果：

   - 评测结果会保存在 `./eval_results` 目录中
   - 可视化图表会保存在 `./visualizations` 目录中
   - 优化建议报告会保存在 `./eval_results/optimization_report.md` 文件中

## 项目结构

```
rag-evaluation-platform/
├── data/              # 测试数据集
│   └── dataset.json   # 默认测试数据集
├── eval_results/      # 评测结果
├── rag/               # RAG系统
│   ├── data/          # RAG系统使用的文档
│   ├── faiss_db_m3e/  # 向量数据库
│   ├── rag_observe.py # RAG系统入口
│   └── rag_with_m3e.py # RAG系统实现
├── visualizations/    # 可视化结果
├── comparison_engine.py # 配置对比功能
├── config.json        # 配置文件
├── config_manager.py  # 配置管理模块
├── evaluation_engine.py # 评测引擎
├── main.py            # 主入口
├── metrics_config.py  # 评测指标配置
├── optimizer.py       # 优化建议生成器
├── rag_client.py      # RAG客户端
└── visualizer.py      # 可视化工具
```

## 评测指标

本平台使用以下评测指标：

### 检索指标
- **Contextual Relevancy**：检索到的上下文与查询的相关程度
- **Contextual Precision**：检索到的相关上下文占所有检索上下文的比例
- **Contextual Recall**：检索到的相关上下文占所有相关上下文的比例

### 生成指标
- **Answer Relevancy**：生成的答案与查询的相关程度
- **Faithfulness**：生成的答案与检索上下文的忠实程度
- **Hallucination**：生成答案中包含幻觉（虚构信息）的程度

### 质量指标
- **Toxicity**：生成答案中包含有害或冒犯性内容的程度
- **Bias**：生成答案中存在的偏见程度

### 性能指标
- **Latency**：从查询到响应的总延迟时间（毫秒）
- **Token Efficiency**：答案token数占总token数的比例
- **Cost**：单次查询的估算成本（美元）

## 优化建议

评测完成后，系统会生成详细的优化建议报告，包括：

1. **指标概览**：所有评测指标的得分和通过率
2. **优化建议**：针对每个低分指标的具体优化建议
3. **失败案例分析**：详细分析失败的测试案例
4. **优化行动计划**：短期、中期和长期的优化行动计划

## 自定义RAG系统

如果你想使用自己的RAG系统进行评测，你需要：

1. 在 `rag` 目录中实现你的RAG系统
2. 在 `config.json` 文件中配置你的RAG系统：

```json
{
  "rag": {
    "type": "local",
    "local_function": "your_module.your_function"
  }
}
```

或者，你也可以实现一个RAG系统客户端，继承 `RAGSystemClient` 类，并在 `main.py` 中使用它。

## 故障排除

### 常见问题

1. **API密钥错误**：确保你在 `config.json` 中设置了正确的API密钥，或者通过环境变量设置了API密钥。

2. **RAG系统连接失败**：确保你的RAG系统正常运行，并且在 `config.json` 中配置了正确的RAG系统路径。

3. **评测结果为空**：确保你的测试数据集格式正确，并且包含有效的测试用例。

4. **可视化图表生成失败**：确保你安装了所有必要的依赖，特别是 `matplotlib` 和 `seaborn`。

## 贡献

欢迎贡献代码、报告问题或提出建议！

## 许可证

本项目采用 MIT 许可证。
