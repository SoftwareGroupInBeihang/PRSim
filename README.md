# PRSim

“基于改动树检索的拉取请求描述生成方法”的源代码。

## 使用

### 训练

**从头开始训练**

```bash
python3 -m prsim.prsim train \
    --param-path params.json
```

**在现有模型基础上继续训练**

```bash
python3 -m prsim.prsim train \
    --param-path params.json \
    --model-path "models/<model_name>/model/<model file>"
```

其中`<model_name>`是参数指定的模型名称，`<model file>`是希望继续训练的模型检查点文件名。

### 验证

```bash
python3 -m prsim.prsim select_model \
    --param-path params.json \
    --model-pattern "models/<model_name>/model/model_{}_" \
    --start-iter 1000 \
    --end-iter 26000
```

其中`<model_name>`是参数指定的模型名称。

### 测试

```bash
python3 -m prsim.prsim decode \
    --param-path params.json \
    --model-path "models/<model_name>/model/<best model>" \
    --ngram-filter 1
```

其中`<model_name>`是参数指定的模型名称，`<best model>`是验证阶段得分最高的模型文件名。
