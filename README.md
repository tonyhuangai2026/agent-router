# Qwen3-1.7B 下一轮分类器（写操作标志 + token 长度）

把 **Qwen3-1.7B** 用 LoRA 在 SageMaker 上微调成一个轻量的「预判分类器」。给定一段对话上下文，它预测**下一个** assistant 回合的两个属性，并只输出一个紧凑 JSON：

```json
{"w":1,"t":512}
```

- `w` ∈ {0,1} —— 下一回合是否包含**写/改类工具调用**（二分类）
- `t` ∈ ℤ⁺ —— 该回合的**输出长度（token 数）**（回归，在 log 空间评分）

**用途**：在昂贵的下游大模型调用**之前**先跑一遍的廉价路由/守门器，用于引擎选型、容量预估等决策。

> ⚠️ **仓库里的 `demo_data.jsonl` 是 mock 样本数据**，仅用于把流水线端到端跑通——它产出的指标没有统计意义。请把 `--input` 指向你自己的真实数据后重跑，**代码无需改动**。

---

## 1. 架构总览

```
 输入数据 (JSONL 文件 或 .json 目录)
        │
        ▼
 data/prepare_data.py  ──用──▶  src/labeling.py     (共享: 写类工具分类法、
        │                                            规范 token 计数、
        │                                            conversation_id、上下文渲染)
        ▼
 data/prepared/{train,val,test}.jsonl + data_stats.json   (展开、打标签、按对话 8/1/1 切分)
        │
        ▼
 launch_sagemaker.py + config.yaml          (HuggingFace Estimator → 上传 S3 → fit())
        │
        ▼
 src/train.py  (SageMaker, HF PyTorch DLC, ml.g5.2xlarge)   LoRA SFT, prompt 掩码损失
        │                                                    训完后在同卡 GPU 顺带评测
        ▼
 s3://<bucket>/qwen-classifier/output/.../model.tar.gz
   (LoRA adapter + tokenizer + run_config.json + eval/{metrics.json,图})
```

**流水线三个阶段：**

1. **准备**：把每条原始记录的 `messages` 历史展开成「上下文 → 下一回合标签」样本（每个 assistant 回合一条）；自动从 `response` 反推标签 `w`（写标志）和 `t`（规范 tokenizer 长度）；派生 `conversation_id`；去重；**按对话**分组切分 train/val/test = 8/1/1（防止同一对话泄漏到不同集合）。
2. **训练**：在 SageMaker 上对 Qwen3-1.7B 做 causal-LM **LoRA 微调**，使用 **prompt 掩码**（loss 只算 completion 部分）。
3. **评测**：对测试集跑真实 `generate()`，输出写分类指标、长度回归指标、格式合法率、真实延迟。**默认在训练作业内、训完后于同一块 GPU 上直接评测**（`run_eval_in_job=true`），所以推理是在 SageMaker GPU 上测的，不是本地 CPU。`evaluate.py` 也能独立运行。

### 仓库结构

```
qwen_classifier/
├── data/
│   ├── prepare_data.py          # 展开、打标签、派生 conversation_id、切分
│   └── prepared/                # train/val/test.jsonl + data_stats.json
├── src/
│   ├── labeling.py              # 写类工具分类法 + 规范 token 计数 + conversation_id（共享）
│   ├── train.py                 # SageMaker 入口: LoRA SFT, prompt 掩码损失, 训内评测
│   ├── evaluate.py              # 批量推理 → metrics.json + 图（含真实延迟）
│   └── requirements.txt         # 训练作业内安装的依赖
├── launch_sagemaker.py          # Estimator + 凭证/角色/桶自动探测 → fit() 或打印运行手册
├── requirements-launch.txt      # 本地提交机所需依赖
├── config.yaml                  # 所有配置（model id、镜像、LoRA、实例、S3、角色、监控正则）
├── run_all.sh                   # 离线演示: prepare → dry-run launch → synthetic eval
└── README.md                    # 本文件
```

---

## 2. 快速开始（离线，一条命令）

无需 AWS、无需训练好的模型，端到端跑 **prepare → launch `--dry-run` → evaluate `--synthetic`**，验证整条流水线接线正确：

```bash
bash run_all.sh
```

它**不会**提交真实 SageMaker 作业、也不做真实推理（那些需要 AWS / 训练好的产物，见 §5、§6）。

---

## 3. 依赖

有意分成**两套**依赖：

| 文件 | 安装位置 | 用途 |
|---|---|---|
| `src/requirements.txt` | **SageMaker 训练作业内**（随 `source_dir` 打包） | 真实训练 |
| `requirements-launch.txt` | **本地提交机** | 提交真实作业 + 离线演示 |

真实提交前：

```bash
pip install -r requirements-launch.txt   # pyyaml + boto3 + sagemaker SDK
```

> `launch_sagemaker.py --dry-run` 懒加载 `sagemaker` SDK，所以 dry-run 校验**无需** SDK（只需 `pyyaml`）。

---

## 4. 分阶段命令（可直接复制）

除注明外，命令都在 `qwen_classifier/` 目录下运行。

### 4.1 准备数据

展开 + 打标签 + 派生 `conversation_id` + 去重 + 按对话 8/1/1 切分：

```bash
python data/prepare_data.py --input <你的数据>
```

**`--input` 支持两种形式：**

- **JSONL 文件** —— 每非空行一条记录（如 `demo_data.jsonl`）。
- **目录** —— 目录下**每个 `.json` 文件算作一条记录**（即 JSONL 的一行）。按文件名排序处理、不递归子目录；若某个 `.json` 本身是 JSON 数组，则数组里每个元素各算一条记录。

```bash
# JSONL 文件:
python data/prepare_data.py --input /path/to/data.jsonl --outdir data/prepared

# 或一个 .json 目录:
python data/prepare_data.py --input /path/to/json_dir/ --outdir data/prepared
```

其它参数（括号内为默认值）：`--max-len 4096`、`--seed 42`、`--val-frac 0.1`、`--test-frac 0.1`、`--dual-use-as-write`。

常用覆盖：

- `--no-dual-use-as-write` —— 把 `run_terminal_cmd` / `bash` 当作**读**（`w=0`），而非写（见 §7）。
- `--conversation-id-field <字段名>` —— 如果你的数据已经有真实的对话/会话分组字段，直接用它分组，而不是派生 id。

每行输出含：`prompt`、`completion`（紧凑 `{"w":..,"t":..}`）、`w`、`t`、`conversation_id`、`session_id`、`usage_output_tokens`（仅参考列，**绝不**作为训练目标）。

### 4.2 训练 —— 本地冒烟（无需下载 1.7B 即可验证训练循环）

`src/train.py` 是 SageMaker 入口，但可以用一个**极小模型**在本地 CPU 端到端验证训练循环：

```bash
python src/train.py \
    --model_id trl-internal-testing/tiny-Qwen3ForCausalLM \
    --max_steps 2 --per_device_batch 1 --epochs 1 \
    --output_dir /tmp/smoke
```

它会把 LoRA adapter + tokenizer + `run_config.json` 写到 `/tmp/smoke`。真实的全量训练在 SageMaker 上跑（§4.3）。

§3.3 的全部超参都暴露为 CLI 参数：`--model_id`、`--max_len`、`--lora_r`、`--lora_alpha`、`--lora_dropout`、`--lr`、`--epochs`、`--per_device_batch`、`--grad_accum`、`--warmup_ratio`、`--seed`、`--bf16`、`--gradient_checkpointing`。在 SageMaker 上这些值来自 `config.yaml` 的 `hyperparameters:` 块。数据目录默认走 SageMaker 通道（`SM_CHANNEL_TRAIN` / `SM_CHANNEL_VALIDATION`），模型目录默认 `SM_MODEL_DIR`（`/opt/ml/model`）。

### 4.3 在 SageMaker 上启动

**Dry-run（离线校验计划，绝不提交）** —— 无需 AWS、无需 SDK（只需 `pyyaml`）：

```bash
python launch_sagemaker.py --dry-run
```

打印完整作业计划：DLC 镜像、实例类型、entry_point / source_dir、S3 输入输出路径、所有透传的超参。

**真实提交（自动探测凭证 → 提交，否则打印运行手册）：**

```bash
python launch_sagemaker.py          # 提交后立即返回
python launch_sagemaker.py --wait   # 提交并实时打印训练日志
```

逻辑：若 `aws sts get-caller-identity` 成功 **且** 能解析出执行角色（env `SAGEMAKER_ROLE` → `config.yaml` 的 `aws.execution_role` → IAM 自动发现）**且** S3 桶可用/可建，就调 `estimator.fit()` 提交**真实**作业；否则打印精确的运行手册并 exit 0（不会让流水线崩溃）。

### 4.4 评测

**默认评测在训练作业内自动完成**（同卡 GPU，训完即跑，`config.yaml` 里 `run_eval_in_job=true`），结果随 `model.tar.gz` 输出到 `eval/`。一般无需手动运行 `evaluate.py`，但它也能独立跑：

```bash
# 合成冒烟（离线，无需模型）—— 仅验证指标/报告/图代码，latency 标记 synthetic=true、非交付:
python src/evaluate.py --synthetic --test_file data/prepared/test.jsonl --report_dir /tmp/eval_synthetic

# 对下载的真实产物评测:
aws s3 cp s3://<bucket>/qwen-classifier/output/<job-name>/output/model.tar.gz /tmp/model.tar.gz
mkdir -p /tmp/model && tar -xzf /tmp/model.tar.gz -C /tmp/model
python src/evaluate.py --model_dir /tmp/model --test_file data/prepared/test.jsonl \
    --report_dir /tmp/eval_real --batch_size 8 --max_new_tokens 16
```

`--device` 默认 `auto`（有 GPU 用 GPU）。CPU 也能跑但每样本远慢于 GPU，优先用训内 GPU 评测。

---

## 5. 输出落点

| 输出 | 位置 |
|---|---|
| 准备好的数据 | `data/prepared/{train,val,test}.jsonl` + `data_stats.json` |
| S3 上的数据（launcher 上传） | `s3://<bucket>/qwen-classifier/data/{train,validation,test}/` |
| **训练好的模型产物** | `s3://<bucket>/qwen-classifier/output/<job-name>/output/model.tar.gz` |
| **评测结果**（GPU，训内） | 打包进 `model.tar.gz` 的 `eval/` 子目录：`metrics.json` + 两张图 |

`model.tar.gz` 含 **LoRA adapter + tokenizer + `run_config.json`**（记录所用全部超参）**+ `eval/` 评测结果**。

> **S3 桶**：`config.yaml` 的 `s3.bucket` 留空时，launcher 用该账号/区域的 SageMaker 默认桶 `sagemaker-<region>-<account>`。填 `s3.bucket` 可覆盖。

---

## 6. SageMaker 运行手册

`config.yaml` 里已配好的默认值：

| 项 | 值 |
|---|---|
| **区域** | `us-east-1`（按需改 `config.yaml` 的 `aws.region`） |
| **账号** | `<YOUR_AWS_ACCOUNT_ID>` —— 填你自己的 12 位账号 ID |
| **执行角色 ARN** | `arn:aws:iam::<YOUR_AWS_ACCOUNT_ID>:role/service-role/AmazonSageMaker-ExecutionRole-XXXX` —— 用你账号里的 SageMaker 执行角色 |
| **实例** | `ml.g5.2xlarge`（单卡 A10G 24GB） |
| **镜像** | HuggingFace PyTorch 训练 DLC，**transformers 4.51** —— `763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training:2.6-transformers4.51-gpu-py312-cu126-ubuntu22.04`（这是 AWS 官方公开 DLC 账号，全区域通用；**Qwen3 需要 transformers ≥ 4.51**） |

### 6.1 执行角色解析顺序

1. env `SAGEMAKER_ROLE`（最高优先级）
2. `config.yaml` → `aws.execution_role`
3. IAM 自动发现名字含 `AmazonSageMaker-ExecutionRole` 的角色

```bash
# 用你自己账号里的 SageMaker 执行角色 ARN:
export SAGEMAKER_ROLE=arn:aws:iam::<YOUR_AWS_ACCOUNT_ID>:role/service-role/AmazonSageMaker-ExecutionRole-XXXX
# 不知道 ARN 时查询:
aws iam list-roles --query "Roles[?contains(RoleName,'AmazonSageMaker-ExecutionRole')].Arn" --output text
```

调用者（不只是角色）还需要 `sagemaker:CreateTrainingJob` 和 `iam:PassRole` 权限。

### 6.2 成本与时长

- `ml.g5.2xlarge` 按需价 ≈ $1.5/小时（us-east-1，以当前 AWS 价为准）。
- 单次墙钟主要花在容器启动 + 基座模型下载（几分钟）+ 训练本身（随 `样本数 × epochs` 线性增长）。
- `compute.max_run_seconds` 是超时上限（默认配成 3600s=1h），作业超时自动停，防止跳表烧钱。

### 6.3 监控 loss 曲线（CloudWatch）

训练作业会把 **loss 曲线实时推到 CloudWatch**，无需 `wandb`/`tensorboard`、无需改 `train.py`。launcher 给 Estimator 传 `metric_definitions`（正则），SageMaker 按正则从作业日志里抓数，发布到命名空间 **`/aws/sagemaker/TrainingJobs`**。正则在 `config.yaml` 的 `metrics.metric_definitions`（launcher 内有兜底）：

| 指标 | 含义 |
|---|---|
| `train:loss` | 每步训练 loss |
| `eval:loss` | 每个 epoch 的验证 loss |
| `learning_rate` | 学习率（含科学计数） |
| `epoch` | 训练进度 |

正则匹配 HF Trainer `PrinterCallback` 每 `logging_steps=5` 打印的 dict 行，兼容 transformers 4.51（无引号）和 5.x（带引号）两种格式。

实时查看：**SageMaker 控制台 → 你的训练作业 → Metrics 标签页**，或 CLI：

```bash
# 列出该作业发布的指标:
aws cloudwatch list-metrics --namespace /aws/sagemaker/TrainingJobs \
  --dimensions Name=TrainingJobName,Value=<job-name> --region us-east-1

# 拉取 train:loss 数据点:
aws cloudwatch get-metric-statistics --namespace /aws/sagemaker/TrainingJobs \
  --metric-name train:loss --dimensions Name=TrainingJobName,Value=<job-name> \
  --start-time <ISO8601> --end-time <ISO8601> --period 60 \
  --statistics Average Minimum Maximum --region us-east-1
```

也可以 `python launch_sagemaker.py --wait` 直接 tail 原始 loss 日志行。

### 6.4 升级 GPU 实例 → `ml.g6e.2xlarge`

当前用 `ml.g5.2xlarge`（A10G 24GB），因为它是本账号/区域唯一有配额（=1）的 GPU 训练实例。更强的 **`ml.g6e.2xlarge`**（L40S 48GB）当前**训练配额为 0**，作业会被 `ResourceLimitExceeded` 拒绝，**需先向 AWS 申请提额**。

提额后切换步骤：

1. **AWS 控制台 → Service Quotas → AWS services → "Amazon SageMaker"**。
2. 找到配额 **`ml.g6e.2xlarge for training job usage`**（CLI 查询）：
   ```bash
   aws service-quotas list-service-quotas --service-code sagemaker --region us-east-1 \
     --query "Quotas[?contains(QuotaName,'g6e.2xlarge for training job usage')].[QuotaName,Value]" --output table
   ```
3. **Request increase** → 设为 **`>= 1`** 提交（单卡通常审批较快）。
4. 批准后，在 `config.yaml` 改：
   ```yaml
   compute:
     instance_type: "ml.g6e.2xlarge"
   ```
   其它都不用动。48GB 的 L40S 还能把 `hyperparameters.max_len` 调回 4096、或加大 `per_device_batch`。CloudWatch 监控保持不变。

---

## 7. 写类工具分类法 & 双用途策略

写标志 `w` 在 `src/labeling.py` 里派生：某回合只要有任一 `tool_use` 块的 `name` 落在**写类集合**就 `w=1`：

```
WRITE_TOOLS = {
    edit_file, search_replace, delete_file, reapply, create_new_file,
    write, apply_patch, str_replace_editor, run_terminal_cmd
}
```

**双用途策略**：`run_terminal_cmd` / `bash` 可能改状态、也可能不改，归为双用途集合：

```
DUAL_USE_TOOLS = { run_terminal_cmd, bash }
```

**默认把双用途工具算作写**（`dual_use_as_write=True`），可审计、可逆：

- `is_write_response(content, dual_use_as_write=True)` → 写集合 = `WRITE_TOOLS ∪ DUAL_USE_TOOLS`（`bash`-only 回合算**写**）
- `is_write_response(content, dual_use_as_write=False)` → 写集合 = `WRITE_TOOLS − DUAL_USE_TOOLS`（`bash`-only 回合算**读**）

**如何覆盖：**

```bash
# 数据准备时翻转整批标签（把 run_terminal_cmd/bash 当读 w=0）:
python data/prepare_data.py --no-dual-use-as-write
```

要改分类法本身（增删工具），编辑 `src/labeling.py` 顶部的 `WRITE_TOOLS` / `DUAL_USE_TOOLS`——数据准备和评测都 import 这些常量，一处改全局生效。

---

## 8. 设计要点与限制

- **`src/requirements.txt` 为支持 Qwen3 的 DLC 钉版本**：`transformers>=4.51,<4.53`、`peft>=0.15,<0.18`、`trl>=0.16,<0.20`、`accelerate>=1.6,<2.0`、`datasets>=3.5,<4.0`。**Qwen3 必须 transformers ≥ 4.51**，所以 `config.yaml` 的 `image.image_uri` 指向 4.51 DLC。改 DLC 镜像 tag 时同步回顾这些钉版本。
- **`demo_data.jsonl` 是 mock 数据**，请换成你自己的真实数据（同 schema 的更大 JSONL，或一个 `.json` 目录）后重跑，代码无需改动。指标质量随数据量与类别均衡度提升——要让指标有统计意义，需用足够多的数据和足够大、均衡的测试集。
- **输入 schema**：每条记录是一次 LLM 请求/响应轨迹；流水线读 `messages` 历史和 `response`，并从规范化的根（首条 user）prompt 派生 `conversation_id`。若数据已有真实分组字段，用 `--conversation-id-field <字段名>` 直接分组。
- **无实时推理端点 / 自动扩缩**：评测用批量加载（训内 GPU 或独立），不部署常驻 SageMaker 端点。
- **无多卡 / 分布式训练**：1.7B LoRA 单卡 A10G 足够，`instance_count: 1`。
- **长度目标用规范 tokenizer 计数，不用 API usage**：`usage.output_tokens` 通常比 tokenizer 计数大、且只在部分回合存在，因此**绝不**作为训练/评测目标——它仅作为参考列 `usage_output_tokens` 保留。评测真值与训练用**同一个**规范 `t`，无口径不一致。
- **`conversation_id` 是派生的**（来自规范化的根 prompt），**不是** `session_id`：因为 `session_id` 不能可靠标识一个对话。`data_stats.json` 会断言三个切分之间 `conversation_id` **零重叠**（防泄漏）。

---

## 9. 文件 → 用途速查

| 文件 | 用途 |
|---|---|
| `src/labeling.py` | 共享：写类分类法、规范 token 计数、`conversation_id`、上下文渲染 |
| `data/prepare_data.py` | 展开 + 打标签 + 切分 → `data/prepared/*` + `data_stats.json`（支持 JSONL 文件或 .json 目录） |
| `src/train.py` | SageMaker 训练入口：LoRA SFT、prompt 掩码损失、训内 GPU 评测 → `/opt/ml/model` |
| `src/evaluate.py` | 批量推理 → `metrics.json` + 图（含真实延迟）；训内运行 + 独立运行 |
| `launch_sagemaker.py` | HuggingFace Estimator + 凭证自动探测 → `fit()` 或运行手册；CloudWatch 监控正则 |
| `config.yaml` | 启动的唯一配置源（model id、镜像、max_len、LoRA、实例、S3、角色、监控） |
| `run_all.sh` | 离线演示：prepare → dry-run launch → synthetic eval |
