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

- `--no-dual-use-as-write` —— 把 `run_terminal_cmd` / `bash` 当作**读**（`w=0`），而非写（见 §8）。
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

### 6.5 Blackwell 新卡（`ml.g7e.*`）—— 需要专用镜像

> ⚠️ **`ml.g7e.*` 不能直接用默认 DLC 镜像。** g7e 用的是 **NVIDIA RTX PRO 6000 Blackwell** GPU（计算能力 **sm_120**）。默认镜像（PyTorch 2.6 / CUDA 12.6，§6 表格那个）的 CUDA kernel 只覆盖到 sm_90（Hopper），**没有 sm_120 的 kernel**，在 g7e 上一启动就报：
> ```
> RuntimeError: CUDA error: no kernel image is available for execution on the device
> ```
> 这**不是代码 bug**，是「卡太新、镜像里的 PyTorch 太老」。Blackwell 需要 **CUDA 12.8 + PyTorch ≥ 2.7**（cu128 wheel 是首个带 sm_120 kernel 的稳定版）。

**解决：用仓库提供的 Blackwell 自建镜像 `docker/Dockerfile.blackwell`**（base = `nvidia/cuda:12.8.1` + 安装 `torch==2.7.* (cu128)`），build 后推到你的 ECR，再让 SageMaker 用它：

```bash
ACCOUNT=<你的AWS账号>; REGION=us-east-1
aws ecr create-repository --repository-name qwen-classifier --region $REGION 2>/dev/null || true
aws ecr get-login-password --region $REGION | docker login --username AWS \
    --password-stdin $ACCOUNT.dkr.ecr.$REGION.amazonaws.com

# 在仓库根目录 build（建议在带 Blackwell 卡的机器上 build+自测）
docker build -f docker/Dockerfile.blackwell -t qwen-classifier:blackwell .
docker tag qwen-classifier:blackwell $ACCOUNT.dkr.ecr.$REGION.amazonaws.com/qwen-classifier:blackwell
docker push $ACCOUNT.dkr.ecr.$REGION.amazonaws.com/qwen-classifier:blackwell
```

然后改 `config.yaml`：

```yaml
compute:
  instance_type: "ml.g7e.2xlarge"        # 同样需先在 Service Quotas 申请 g7e 配额
image:
  image_uri: "<ACCOUNT>.dkr.ecr.us-east-1.amazonaws.com/qwen-classifier:blackwell"
```

**上 g7e 前先验证镜像认得这块卡**（在 g7e 实例上跑）：
```python
import torch
print(torch.__version__, torch.version.cuda)
print(torch.cuda.get_device_capability())  # g7e 期望 (12, 0)
print("sm_120" in "".join(torch.cuda.get_arch_list()))  # 应为 True
```

> **如实说明**：`Dockerfile.blackwell` 依据 NVIDIA/PyTorch 官方 Blackwell 支持矩阵（CUDA 12.8 / PyTorch 2.7 cu128）编写，但**未能在本仓库的构建机上做 GPU 实测**（无 Blackwell 卡）——请在 g7e 上用上面的命令确认后再正式训练。
>
> **更省事的替代**：1.7B LoRA 这种小模型在 **`ml.g5.2xlarge`（A10G）上几分钟就训完**，没必要上 Blackwell。除非客户只有 g7e 配额或要训更大模型，否则直接用 §6 默认的 g5 + 现成 DLC 镜像最稳。

---

## 7. 本地容器训练（不依赖 SageMaker）

除了 §6 的 SageMaker 方案，仓库还提供一套**纯本地、容器化**的训练路径：一份 Docker 镜像跑通 `prepare → train → evaluate` 全链路，复用 §4 的同一套 `prepare_data.py` / `train.py` / `evaluate.py` 代码（不改训练逻辑），**不需要 AWS / SageMaker**。

> 两套方案并存、互不影响，按场景择一即可：
> - **本地容器**（本节）：手头有一台机器（最好带 NVIDIA GPU，无卡也能在 CPU 上冒烟），想离线/内网把流水线端到端跑通、产物全部落本地磁盘；不想碰 AWS。
> - **SageMaker**（§6）：要做**真实的全量训练**、用云上 A10G/L40S GPU、把产物存 S3、用 CloudWatch 看 loss 曲线。1.7B 全量微调请走这条。
>
> 容器里的 `train.py` / `evaluate.py` 是同一份代码：缺少 `SM_CHANNEL_*` / `SM_MODEL_DIR` 等 SageMaker 环境变量时自动回退到本地目录默认值，所以无需为本地容器改任何代码。

相关文件：`docker/Dockerfile`、`docker/entrypoint.sh`、`docker-compose.yml`、`.env.example`（均在仓库根 / `docker/` 下）。

### 7.1 镜像与阶段

镜像基于 `pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime`（与 SageMaker 同栈：torch 2.6 / CUDA 12.6）。该 base 自带 GPU 版 torch，**无 GPU 时同一个 torch 透明回退到 CPU**；镜像刻意不重装 torch（`src/requirements.txt` 不含 torch），只在其上加 LoRA-SFT 依赖 + `numpy` / `matplotlib`（供 evaluate 画图）。

> **首次 build 会从 Docker Hub 拉取 base 镜像**（CUDA runtime，体积约 6GB+，含下载耗时）；之后命中本地缓存即很快。

`docker/entrypoint.sh` 按第一个参数分发阶段（默认 `all`），并把环境变量超参拼到对应 python 命令：

| 阶段 | 容器内动作 | 读 | 写 |
|---|---|---|---|
| `prepare` | `python data/prepare_data.py --input $INPUT_FILE --outdir /work/prepared` | `/data` | `/work/prepared` |
| `train` | `python src/train.py --train_dir /work/prepared --val_dir /work/prepared --output_dir /work/model [超参]` | `/work/prepared` | `/work/model` |
| `evaluate` | `python src/evaluate.py --model_dir /work/model --test_file /work/prepared/test.jsonl --report_dir /work/report` | `/work/model`,`/work/prepared` | `/work/report` |
| `all` | 依次 `prepare → train → evaluate`，任一阶段失败即非零退出 | | |

### 7.2 一键 / 分阶段命令

命令都在 `qwen_classifier/`（仓库根，即 `docker-compose.yml` 所在目录）下运行。首次跑会自动 build 镜像（也可先手动 `docker build -f docker/Dockerfile -t qwen-classifier:local .`）。

**一键全链路**（`prepare → train → evaluate` 在一个容器里依次跑完）：

```bash
docker compose run --rm all
```

**分阶段**（按需单独跑某一阶段；阶段间通过 `./outputs/` 卷传递产物）：

```bash
docker compose run --rm prepare      # data/prepare_data.py  -> ./outputs/prepared
docker compose run --rm train        # src/train.py          -> ./outputs/model
docker compose run --rm evaluate     # src/evaluate.py       -> ./outputs/report
```

> 默认 `MODEL_ID=Qwen/Qwen3-1.7B`。直接这么跑 = 用真实 1.7B 基座（需联网拉模型、CPU 上会很慢）。只想验证「接线是否跑通」时，强烈建议用 §7.5 的 tiny 模型 + `MAX_STEPS=2` 冒烟参数。

### 7.3 GPU vs CPU 两种跑法

- **无 GPU（本机默认情形）**：上面的命令**直接就在 CPU 上跑**，无需任何额外配置——base 镜像里的 torch 自动回退 CPU。CPU 上 1.7B 全量训练很慢，仅适合用 tiny 模型冒烟（§7.5）。
- **有 GPU**：先在宿主机装 [`nvidia-container-toolkit`](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) 并重启 Docker，然后任选一种把 GPU 暴露进容器：

  ```bash
  # 方式 A：compose 的 gpu profile（用 train-gpu service，已配 nvidia 设备预留）
  docker compose --profile gpu run --rm train-gpu

  # 方式 B：直接 docker run 挂全部 GPU
  docker run --gpus all --rm \
    -v "$PWD/data:/data:ro" \
    -v "$PWD/outputs/prepared:/work/prepared" \
    -v "$PWD/outputs/model:/work/model" \
    -v "$PWD/outputs/report:/work/report" \
    qwen-classifier:local train
  ```

> ⚠️ **GPU 路径未在本机实测**：开发本机无 NVIDIA GPU（CPU-only），GPU 跑法靠 base 镜像自带的 CUDA torch + compose 的 `deploy.resources.reservations.devices`（nvidia / count all）配置保证，未做真机验证。CPU 全链路已实测跑通（§7.5）。

### 7.4 卷挂载、输入数据与可调超参

**卷挂载**（`docker-compose.yml` 已配，产物全部落宿主机，容器外可见）：

| 宿主机 | 容器内 | 用途 |
|---|---|---|
| `./data` | `/data`（只读 `:ro`） | 原始输入数据 |
| `./outputs/prepared` | `/work/prepared` | prepare 产物：`train/val/test.jsonl` + `data_stats.json` |
| `./outputs/model` | `/work/model` | train 产物：LoRA adapter + tokenizer + `run_config.json` |
| `./outputs/report` | `/work/report` | evaluate 产物：`metrics.json` + `report.md` + 两张图 |

**把输入指向你自己的数据**：默认读 `/data` 整个目录。用 `INPUT_FILE` 指向 `./data` 下的具体文件或子目录（容器内路径，即 `/data/...`）：

```bash
# 指向一个 JSONL 文件（每非空行一条记录）：
INPUT_FILE=/data/your_data.jsonl docker compose run --rm prepare

# 或指向一个目录（目录下每个 .json / .jsonl 文件按 §4.1 规则解析）：
INPUT_FILE=/data/your_dir docker compose run --rm prepare
```

> `--input` 的两种形式与取值规则同 §4.1：JSONL 文件（逐行）、或目录（每个 `.json`/`.jsonl` 文件，`.json` 一文件一记录或一数组、`.jsonl` 逐行）。

**可调超参（不改任何文件）**：`docker-compose.yml` 用 `environment: ${VAR:-默认}` 暴露常用超参，可通过 shell env 或 `.env` 覆盖。**完整可调项与默认值见仓库根的 [`.env.example`](.env.example)**（`cp .env.example .env` 后编辑，compose 会自动加载 `./.env`）：

| 变量 | 作用 | 空值行为 |
|---|---|---|
| `MODEL_ID` | 基座模型 id（HF hub 名或挂载的本地路径） | 默认 `Qwen/Qwen3-1.7B` |
| `EPOCHS` / `MAX_STEPS` / `PER_DEVICE_BATCH` / `MAX_LEN` | train 超参 | 空 → 落到 `train.py` 各自 argparse 默认 |
| `INPUT_FILE` | prepare 输入文件/目录（容器内路径） | 默认 `/data` |
| `EVAL_BATCH_SIZE` / `EVAL_MAX_NEW_TOKENS` | evaluate 参数 | 空 → `evaluate.py` 默认（8 / 16） |
| `PREPARE_FORCE_FALLBACK` / `EVAL_SYNTHETIC` | 离线开关（见 §7.6） | 空 → 正常（联网）模式 |

> 机制：env 为空时落到脚本自带默认；entrypoint 只在 env **非空**时才追加对应 `--flag`。

### 7.5 CPU + tiny 模型冒烟（已实测跑通的全链路）

无 GPU 时，用一个**极小模型** + `MAX_STEPS=2` 在 CPU 上端到端验证「prepare→train→evaluate 接线是否正确」。这正是本仓库在 CPU-only 机器上实测过的命令（容器可联网拉 tiny 模型时走真实链路）：

```bash
# 0) 准备一个小输入（示例：取 mock 数据前 30 行）
mkdir -p data outputs/prepared outputs/model outputs/report
head -n 30 demo_data.jsonl > data/demo.jsonl

# 1) 一次性导出冒烟超参（也可写进 .env）
export MODEL_ID=trl-internal-testing/tiny-Qwen3ForCausalLM
export MAX_STEPS=2 PER_DEVICE_BATCH=1 EPOCHS=1
export INPUT_FILE=/data/demo.jsonl

# 2) 全链路（或把 all 换成 prepare / train / evaluate 分阶段跑）
docker compose run --rm all
```

跑完后，宿主机 `./outputs/` 即包含三阶段产物（容器以 root 运行，文件归属 root，见 §7.7）：

```
outputs/prepared/{train,val,test}.jsonl + data_stats.json   # prepare
outputs/model/   adapter_config.json + adapter_model.safetensors + tokenizer* + run_config.json   # train
outputs/report/  metrics.json + report.md + confusion_matrix.png + length_scatter.png             # evaluate
```

> ⚠️ **tiny 模型 + `MAX_STEPS=2` 的指标完全没有意义**（和 §2 / §9 对 `demo_data.jsonl` 的口径一致）：这只是验证流水线接线、产物落盘、卷挂载是否正常，**不是**一次有效训练。tiny 未训练模型几乎必然输出非法 JSON（`output_format_validity` 接近 0、写分类/长度指标为 `n/a`），属预期现象。要拿有意义的指标，请换成真实基座 + 真实数据，并优先用 GPU（本地 §7.3 或云上 §6）。

### 7.6 离线 / 内网开关

容器**联网**时，prepare 拉规范 tokenizer、train/evaluate 拉基座模型，走真实链路（上面 §7.5 即在联网下实测通过）。**无外网**（内网/离线）时，用以下开关跑通代码路径（产物结构与真实模式一致，但内容是骨架/占位，**不可作为交付**）：

| 阶段 | 开关 | 效果 |
|---|---|---|
| prepare | `PREPARE_FORCE_FALLBACK=1` | 用轻量 tokenizer、不下载任何模型，照常完成展开/打标签/切分 |
| evaluate | `EVAL_SYNTHETIC=1` | 走 `evaluate.py --synthetic`：不加载模型，离线产出报告骨架，latency 标记 `synthetic=true`、非交付 |

```bash
# 离线 prepare（无模型下载）
PREPARE_FORCE_FALLBACK=1 INPUT_FILE=/data/demo.jsonl docker compose run --rm prepare

# 离线 evaluate（无需训练好的模型）
EVAL_SYNTHETIC=1 docker compose run --rm evaluate
```

> 说明：
> - 离线开关是 `PREPARE_FORCE_FALLBACK` / `EVAL_SYNTHETIC`（**不是** `LABELING_FORCE_FALLBACK`——后者只对 `labeling.py` 的 `__main__` 自测有效，对 prepare/evaluate 无效）。
> - **`train` 没有离线骨架模式**：真实训练必须能拿到基座模型（联网拉，或把 `MODEL_ID` 指向已挂载进 `./data` 的本地模型目录）。
> - **evaluate 真实模式需联网**拉基座模型（除非用本地模型路径）；只有 `--synthetic` 才完全离线。

### 7.7 容器以 root 运行：产物文件归属

容器默认以 **root** 运行，因此写到宿主机 `./outputs/` 的文件归属 `root:root`，用普通用户删除/改动时可能需要 `sudo`。两种处理：

```bash
# 事后清理（需要 sudo，因为文件属 root）：
sudo rm -rf outputs/prepared/* outputs/model/* outputs/report/*

# 或让容器以你的宿主 uid:gid 运行，产物直接归属当前用户（按需逐条加 user:）：
docker compose run --rm --user "$(id -u):$(id -g)" prepare
```

> 用 `--user` 映射时，需确保宿主机 `./outputs/` 子目录对该 uid 可写（本仓库 compose 已预设挂载点，普通情况下可写）。

---

## 8. 写类工具分类法 & 双用途策略

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

## 9. 设计要点与限制

- **`src/requirements.txt` 为支持 Qwen3 的 DLC 钉版本**：`transformers>=4.51,<4.53`、`peft>=0.15,<0.18`、`trl>=0.16,<0.20`、`accelerate>=1.6,<2.0`、`datasets>=3.5,<4.0`。**Qwen3 必须 transformers ≥ 4.51**，所以 `config.yaml` 的 `image.image_uri` 指向 4.51 DLC。改 DLC 镜像 tag 时同步回顾这些钉版本。
- **`demo_data.jsonl` 是 mock 数据**，请换成你自己的真实数据（同 schema 的更大 JSONL，或一个 `.json` 目录）后重跑，代码无需改动。指标质量随数据量与类别均衡度提升——要让指标有统计意义，需用足够多的数据和足够大、均衡的测试集。
- **输入 schema**：每条记录是一次 LLM 请求/响应轨迹；流水线读 `messages` 历史和 `response`，并从规范化的根（首条 user）prompt 派生 `conversation_id`。若数据已有真实分组字段，用 `--conversation-id-field <字段名>` 直接分组。
- **无实时推理端点 / 自动扩缩**：评测用批量加载（训内 GPU 或独立），不部署常驻 SageMaker 端点。
- **无多卡 / 分布式训练**：1.7B LoRA 单卡 A10G 足够，`instance_count: 1`。
- **长度目标用规范 tokenizer 计数，不用 API usage**：`usage.output_tokens` 通常比 tokenizer 计数大、且只在部分回合存在，因此**绝不**作为训练/评测目标——它仅作为参考列 `usage_output_tokens` 保留。评测真值与训练用**同一个**规范 `t`，无口径不一致。
- **`conversation_id` 是派生的**（来自规范化的根 prompt），**不是** `session_id`：因为 `session_id` 不能可靠标识一个对话。`data_stats.json` 会断言三个切分之间 `conversation_id` **零重叠**（防泄漏）。

---

## 10. 文件 → 用途速查

| 文件 | 用途 |
|---|---|
| `src/labeling.py` | 共享：写类分类法、规范 token 计数、`conversation_id`、上下文渲染 |
| `data/prepare_data.py` | 展开 + 打标签 + 切分 → `data/prepared/*` + `data_stats.json`（支持 JSONL 文件或 .json 目录） |
| `src/train.py` | SageMaker 训练入口：LoRA SFT、prompt 掩码损失、训内 GPU 评测 → `/opt/ml/model` |
| `src/evaluate.py` | 批量推理 → `metrics.json` + 图（含真实延迟）；训内运行 + 独立运行 |
| `launch_sagemaker.py` | HuggingFace Estimator + 凭证自动探测 → `fit()` 或运行手册；CloudWatch 监控正则 |
| `config.yaml` | 启动的唯一配置源（model id、镜像、max_len、LoRA、实例、S3、角色、监控） |
| `run_all.sh` | 离线演示：prepare → dry-run launch → synthetic eval |
