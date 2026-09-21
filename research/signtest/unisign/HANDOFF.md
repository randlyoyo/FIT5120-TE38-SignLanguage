# 交接文档：Auslan → English 手语翻译（Uni-Sign pose-only 迁移学习）

更新于 2026-09-17。写给接手这个项目的新对话窗口。

新窗口是 Claude Code、并且在同一个项目目录里的话，先读这份，再读 `unisign/README.md`。README 是英文的技术说明，比这份更细。代码都在 `unisign/` 下。

---

## 0. 协作约定（先看）

- **语言**：用户用中文交流，回答用中文。
- **"文字说明 / 文字回答"**：用户这么说，就只用文字解释，不改代码。
- **`.ipynb` 在 VS Code 里开着时，不要改它。** 否则 Colab 扩展会报 `controller is DISPOSED`，笔记本内容还可能被覆盖。要改先请用户关掉那个标签页。
- 用户在 VS Code 里通过 Colab 扩展连接 Colab（A100），Google Drive 有 5 TB 空间。
- 代码改完要重新打包 `unisign_code.tar.gz`（在项目根目录，不含 `*.ipynb`、`*.bak`、`__pycache__`），并提醒用户把 `unisign/` 重新传到 Drive 的 `MyDrive/unisign/`。
- 改完代码要实际跑测试验证，不要只说"应该可以"。

---

## 1. 目标与范围

- 只做 **sign → text**：正面单机位 Auslan 视频 → 英文句子（SLT）。不做手语生成。
- 基座：**Uni-Sign**（ICLR 2025）pose-only checkpoint，许可证 **CC BY-NC 4.0，非商用**，微调出来的模型也继承这一点。
- 基座的预训练语料是中国手语（CSL）和美国手语（ASL），**不含 Auslan**。迁移过来的是运动表征和 encoder-decoder 结构，不是词汇。
- **期望**：只有 45 小时连续数据、67 位手语者，BLEU 大概率是个位数。作为参照，YouTube-SL-25 用 1394 小时 ASL 预训练，在 How2Sign 上是 15.4 BLEU。
- **评测**：BLEU-1~4、ROUGE-L（可选 BLEURT-20）。Communication 和 News **分开报**，永远不取平均。另外报测试集相对训练集的 OOV 率。

## 2. 原计划（用户最初给的方案 v1）

1. 打通 Uni-Sign 环境、验证 checkpoint。**已完成**
2. 获取数据。**已完成**：Auslan-Daily 全部到位；MM-WLAuslan 已下载并完成姿态抽取
3. 统一预处理，抽关键点，抽样可视化验证。**Auslan-Daily 与 MM-WLAuslan 已完成**
4. **Arm A 基线**：只用 Auslan-Daily 微调。**已完成**
5. Arm B 和 Arm C，可以并行跑：
   - **Arm B**：分两阶段。第 1 阶段在 MM-WLAuslan 上只训练姿态编码器，时序模块和 mT5 的学习率降到 1/10 或冻结；第 2 阶段在 Auslan-Daily 上全部解冻微调。**顺序不能反。**
   - **Arm C**：只有一个阶段。在 Auslan-Daily 上训练，同时混入低权重的 MM-WLAuslan 辅助损失，权重扫 0.1 / 0.2 / 0.3。
6. **Arm D**（可选）：把孤立词片段拼成伪连续句，用来缓解 OOV。要不要做，看 A/B/C 的 OOV 率。
7. 固化结论，评估要不要上 RGB 分支。

原计划里被代码证伪的几处，见第 4 节。

## 3. 当前状态（2026-09-17）

### 已完成
- **姿态抽取**：Auslan-Daily 共 25,109 段（Communication 14,040，News 11,069），在 Colab A100 上抽完。结果在 Drive `MyDrive/auslan_work/pose/chunk_*.tar`。
- **训练前检查（`verify_pose.py`）已通过**：
  - 一致性：全部片段是同一个 spec、同一个模型、同一个选人策略，指纹 `bc3bb2df0f22948d`，`person=largest`。
  - 质量标记都低于 1%，只是提示，不拦训练。
  - **镜像检测标出 131 段（0.5%）**：Communication 113（0.80%），News 18（0.16%）。按 split 分：train 111、val 8、test 12。这些片段已写进 `auslan_work/excluded.txt`，训练时通过 `data.exclude` 跳过。
  - 镜像片段里画面中有第二个人的比例是 62%，全部片段是 36%。说明一部分是选错了人，其余是人背对或侧身、模型把左右认反了。**判断：是个别片段的问题，不是系统性翻转，排除即可。**
  - 叠加视频在 `auslan_work/overlays_mirrored/`。用户已检查其中的 `ad-communication-video_29_97.mp4`，确认该片段的姿态跟踪明显错误；这支持将这类片段排除，而不是把问题判断为全局镜像。
- **MM-WLAuslan 姿态抽取已完成（2026-09-15）**：
  - `colab_setup.ipynb` 使用 MM-WLAuslan-only manifest，只抽取 `Train`、`Valid`、`Test_STU` 的正面 `kf` 相机，没有复制或读取 Auslan-Daily 视频。
  - manifest 共 51,440 段：Train 38,580、Valid 6,430、Test_STU 6,430；共 3,215 个孤立 gloss。
  - 14 个 worker 在单张 A100 上完成抽取，最终输出为 `51,440/51,440 MM-WLAuslan clips extracted`；新生成的姿态分片写入 Drive `MyDrive/auslan_work/pose/chunk_*.tar`。
  - 最终 gate 已通过：`spec=unisign-pose-v2-verified`，指纹 `bc3bb2df0f22948d`，`rtmlib.Wholebody/lightweight`、`onnxruntime`、`person=largest`；51,440 段中仅有 17 段 `out_of_bounds`（0.0%），没有 `MIRRORED_OR_BACK_VIEW`，排除数为 0。
  - 本次 MM-WLAuslan 检查没有生成新的异常叠加视频；`--overlay-n 131` 只是可视化样本上限，不影响姿态抽取数量。
  - **路径冲突待处理**：本次 MM-WLAuslan 检查仍把质量结果、排除名单和可视化目录写到 `auslan_work/quality.csv`、`auslan_work/excluded.txt`、`auslan_work/overlays_mirrored/`。输出中的 `0 clips excluded` 可能已经覆盖此前 Auslan-Daily 的排除名单；在开始 B/C 训练前必须确认 `excluded.txt` 仍包含 Auslan-Daily 的 131 个异常 UID（训练/验证实际涉及 119 段），不能直接把这次 MM-WLAuslan 的空名单当作联合训练排除名单。
- **训练笔记本 `colab_train.ipynb`**：共 9 节。原来的 30 步试跑已按用户要求删除。
  - 第 1–6 节准备环境、模型、数据和配置。
    - 第 6 节的 `LR`、`SEEDS` 决定第 8 节训练哪些 run。3e-5 的基线（种子 0、1）始终列出来做参照，沿用原来的 run 名；其他学习率的 run 名带后缀，比如 `__lr1e-04`。
    - `DECODES` 列出要评估的解码设置。
    - `run_train` 等辅助函数也定义在第 6 节，所以每个会话都要跑。
  - 第 7 节：用 `train.py --eval-only` 对已完成的 run 按每种解码设置重新评估，结果写进 run 目录的 `eval_<tag>/`。第一次运行时，还会用 plain 解码重评基线种子 0，结果必须和它原来的分数一致。
  - 第 8 节：训练 `SEEDS` 里的 run。训完先做 plain 评估，再按 `DECODES` 评估。
  - 第 9 节：所有 run × 所有解码设置的对比表，按子集分开，附基线范围和判定线。
  - 断线后重跑 1–6，再跑 8。第 7、9 节随时可以跑，已完成的会跳过。
- **Arm A 种子 0 已跑完**：起点 `csl_stage1_weight.pth`，20 个 epoch，约 0.3 秒/步，A100 上约 4.5 小时。Colab 上的 transformers 5.16.1 可以正常训练和生成。验证集成绩：
  - Communication（n=792）：BLEU-1 25.22，BLEU-4 3.42，ROUGE-L 13.8
  - News（n=700）：BLEU-1 8.6，BLEU-4 0.68，ROUGE-L 8.31
- **种子 0 的诊断**：
  - 训练 loss 从 13.2 降到约 5.65 后进入平台期，离 label smoothing 0.2 的理论下限 2.99 还很远。
  - Communication 42% 的输出是同一句 `that is great .`，71% 是训练集里的原句。
  - News 64% 的输出陷入重复循环，比如 `a bit of a bit of…`。
  - 对照：把模型输出打乱后再配给别的片段，Communication BLEU-4 从 3.42 降到 1.27。说明模型从姿态里学到的信号很少，主要在输出高频句。
  - 学习率 3e-5 比 Uni-Sign 官方设置（3e-4，4 张 GPU × batch 8 = 32）按 batch 8 换算后还低 2.5–5 倍。判断更像欠拟合，不是数据太少。

- **Arm A 种子 1 也已跑完**，配置相同：
  - Communication：BLEU-1 27.98，BLEU-4 4.05，ROUGE-L 13.67
  - News：BLEU-1 5.13，BLEU-4 0.46，ROUGE-L 6.03
  - 两个种子每个 epoch 的平均 loss 差距都在 ±0.12 以内。
  - **噪声底线**（两个种子之差）：Communication BLEU-4 0.63，News BLEU-4 0.22。用种子 0 的预测做 bootstrap，Communication BLEU-4 的抽样标准差也有 0.62，说明验证集本身的噪声就不小。
  - 种子 1 的输出问题和种子 0 一样，News 更严重：77% 的输出陷入重复循环，平均 37.6 个词，参考译文只有 16 个。

### 官方 Stage3 配置实验（已完成：当前最佳 Arm A 参考）

为了验证旧基线是否因为学习率过低而欠拟合，使用论文官方 Stage3 的优化器配置，在单张 A100 上通过梯度累积模拟官方的有效 batch size。对应的代码和 notebook 是 `train.py`、`colab_train_official.ipynb`，运行名为 `arm_a__csl_stage1_weight__official_stage3__single_a100`。

这次实验和 `colab_train.ipynb` 里的自定义 Arm A 基线/学习率实验不是同一套优化配置。后者的旧基线使用 `lr=3e-5`、普通 batch 8；官方 Stage3 同时使用更高的 `lr=3e-4`、有效 batch 32、`weight_decay=1e-4`、无 warmup 和 cosine decay。因此，下面的提升说明“官方 Stage3 优化组合”有效，不能归因于学习率一个变量。

- 数据：Auslan-Daily，训练集 21,998 段，验证集 1,492 段；仍然排除 `excluded.txt` 中的 119 段训练/验证镜像或疑似反视角片段。
- 初始化：`csl_stage1_weight.pth`。
- 训练：20 个 epoch，单卡 micro-batch 8，`gradient_accumulation_steps=4`，有效 batch size 32，每 epoch 687 次 optimizer update，共 13,740 步。
- 优化器：AdamW，`lr=3e-4`，`weight_decay=1e-4`，`betas=(0.9, 0.999)`，`eps=1e-9`，梯度裁剪 1.0。
- 学习率策略：无 warmup，使用 cosine decay；最后几轮学习率接近 0，这是预期行为，不是训练中断。
- 其他：`label_smoothing=0.2`，`max_length=256`，验证生成使用 4 beams。
- 模型加载检查：`missing=0`、`unexpected=0`；总可训练参数约 587.75M，说明这次确实加载并训练了完整的 Uni-Sign pose-only 模型。

训练过程：

- 第 1 轮已记录 loss 的平均值约 6.69；第 10 轮约 4.87；第 14 轮约 4.63；第 20 轮约 4.51。
- **2026-09-16 重跑了这个 Arm A run**，epoch 0 的平均 loss 约 **6.707**。两个 notebook 的 epoch 0 判据以 6.707 为准。重跑用的是 fp32 还是 bf16 尚未确认，需要补记。
- 最后若干日志点约在 4.35～4.70 之间波动，训练后半段已经进入平台期。
- 相对于旧基线末期约 5.65，这次末期 loss 下降约 1.1；相对于旧的 3e-5 配置，前期 loss 也明显更低。
- 当前 loss 不能直接与“必须达到 2”画等号。`label_smoothing=0.2` 改变了交叉熵目标，约 2.99 是当前平滑标签目标的理论下界估算，不是论文报告的实验 loss，也不是论文规定的验收线。

普通 beam search 解码结果（验证集）：

| 子集 | n | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | ROUGE-L | looping | hyp/ref length | OOV type |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Communication | 792 | 36.12 | 20.51 | 14.63 | 11.51 | 23.78 | 1.8% | 5.1 / 5.2 | 0.8% |
| News | 700 | 17.39 | 5.96 | 3.22 | 2.17 | 12.90 | 34.6% | 18.6 / 15.9 | 10.2% |

在相同最终权重上额外测试 `no_repeat_ngram_size=3`，这不是重新训练：

| 子集 | BLEU-4（普通） | BLEU-4（nr3） | ROUGE-L（普通） | ROUGE-L（nr3） | looping（nr3） | hyp/ref length（nr3） |
|---|---:|---:|---:|---:|---:|---:|
| Communication | 11.51 | 11.69 | 23.78 | 24.01 | 0.0% | 4.9 / 5.2 |
| News | 2.17 | 2.63 | 12.90 | 13.35 | 0.0% | 17.1 / 15.9 |

结果判断：

- **优化配置有效**：Communication 的 BLEU-4 从旧基线最高 4.05 提升到 11.51，News 从旧基线最高 0.68 提升到 2.17。这个提升远大于此前两个 seed 之间的噪声差距（Communication 0.63，News 0.22）。
- **Communication 已经不再是主要问题**：预测长度与参考长度接近，普通解码的 looping 只有 1.8%，说明模型已经从姿态中学到相当一部分 sign-to-text 对应关系，不是只输出固定高频句。
- **News 是当前主要瓶颈**：News 的 OOV type 为 10.2%，包含较多新闻人名、地名和专有词；普通解码的预测也偏长并有 34.6% 的重复。`nr3` 消除重复后，BLEU-4 提升到 2.63，说明解码重复是其中一个真实因素，但词汇覆盖和领域差异仍未解决。
- **训练后期收益有限**：第 14～20 轮 loss 只从约 4.63 降到约 4.51，并且 cosine 学习率已经接近 0。继续沿用同一个已经衰减到 0 的学习率不会带来明显更新；若要延长训练，应作为新的、记录清楚的实验重新设定学习率，不能直接把它当作本次 20 epoch 结果。
- **当前结论只有一个官方 Stage3 seed**：本次结果足以把它作为当前最好的 Arm A 参考，但还不能给出均值和标准差。由于 GPU unit 限制，近期不继续跑 official `seed=1`；这不影响现在进入 Arm B/C，但最终论文式汇报时要注明这是单 seed 结果。

评估时出现的 SacreBLEU “tokenized period” 提示是输入句子末尾标点格式的警告，不是训练失败；由于旧基线和新实验使用同一个评估脚本，当前相对比较仍然有效。若最终要严格对齐论文报告格式，再统一做 detokenization 后重新报一次。

### 官方 Arm A Stage3 BF16 重跑结果（2026-09-17）

为了检查 BF16 是否能在 A100 上加速，同时保持与官方 Stage3 相同的优化方案，重新训练了一个独立 run。旧的 FP32 官方结果没有被覆盖。

- notebook：`colab_train_official.ipynb`；run：`arm_a__csl_stage1_weight__official_stage3__single_a100__bf16`。
- 设备和环境：NVIDIA A100-SXM4-40GB，`torch 2.11.0+cu128`，`transformers 5.16.1`，CUDA 可用。
- 数据：联合 manifest 共 76,549 段姿态；实际训练仍为 Auslan-Daily train 21,998 段、val 1,492 段；AD gate 排除 131 段，其中 train/val 实际排除 119 段；姿态缺失数为 0。
- 模型加载：`missing=0`、`unexpected=0`；全部 587.75M 参数可训练。
- 训练参数：`lr=3e-4`、`weight_decay=1e-4`、batch size 8、gradient accumulation 4、有效 batch size 32、20 个 epoch、共 13,740 个 optimizer step、gradient clipping 1.0、`label_smoothing=0.2`、4 beams（`colab_train_official.ipynb` 生成的配置是 `num_beams: 4`；此处旧版写成 5 beams 是笔误，2026-09-17 已核对更正）、`max_new_tokens=100`。
- 学习率策略：无 warmup（`warmup_frac=0.0`），cosine decay；最后学习率降到 0。
- 精度：`precision=bf16`。这是 autocast BF16，模型参数和 AdamW 状态仍保持 FP32，不是把整个模型永久转成 BF16。
- 训练从原始 `csl_stage1_weight.pth` 开始，不需要 B1；checkpoint 已保存到 Drive。

BF16 官方 Stage3 的 plain 验证结果：

| 子集 | n | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | ROUGE-L | 不同输出 | looping | hyp/ref 长度 | OOV type |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Communication | 792 | 33.62 | 17.71 | 12.37 | 9.77 | 20.55 | 40.9% | 2.4% | 5.1 / 5.2 | 0.8% |
| News | 700 | 16.85 | 5.57 | 2.85 | 1.82 | 12.33 | 65.1% | 37.9% | 18.4 / 15.9 | 10.2% |

与原 FP32 官方 Stage3 plain 结果比较：

| 子集 | 指标 | FP32 | BF16 | 变化 |
|---|---|---:|---:|---:|
| Communication | BLEU-1 | 36.12 | 33.62 | -2.50 |
| Communication | BLEU-4 | 11.51 | 9.77 | -1.74（约 -15.1%） |
| Communication | ROUGE-L | 23.78 | 20.55 | -3.23（约 -13.6%） |
| News | BLEU-1 | 17.39 | 16.85 | -0.54 |
| News | BLEU-4 | 2.17 | 1.82 | -0.35（约 -16.1%） |
| News | ROUGE-L | 12.90 | 12.33 | -0.57（约 -4.4%） |

结果判断：

- BF16 训练正常完成，没有 NaN，模型完整加载，说明 BF16 AMP 实现和 checkpoint 保存是可用的。
- 这一次 BF16 的验证分数低于 FP32，尤其 Communication 的 BLEU-4 和 ROUGE-L 下降较明显。由于目前只有一个 BF16 seed，不能单独证明下降完全由数值精度造成，但当前用于最终指标的最佳 Arm A 仍应保留 FP32 官方结果。
- News 仍是主要瓶颈：plain 解码的 looping 为 37.9%，平均输出 18.4 词而参考为 15.9 词；OOV type 仍为 10.2%。这说明 BF16 没有解决新闻领域的重复和词汇覆盖问题。
- 速度不能从本次保存的 notebook 输出精确确定，因为训练从 step 13,053 的 checkpoint 续训，输出只记录了最后一个 epoch。最后约 687 个 optimizer step 用时约 13 分钟，按此段外推全程约 4.3 小时；这只能作为粗略估计，不能替代完整 FP32/BF16 wall-clock 对照。
- notebook 的 optional `nr3` 评估尚未保存完整结果；当前 cell 只显示了开始加载和 `--eval-only`，没有最终指标。需要时重新运行 optional 解码 cell，才能判断 `no_repeat_ngram_size=3` 是否像 FP32 一样改善 News。

当前决策：BF16 可以用于后续需要节省显存或尝试加速的搜索实验，但如果最终目标是最高验证/测试分数，暂时不能替代 FP32 官方 Arm A。所有候选搜索必须统一使用 BF16 或统一使用 FP32，不要混用后直接比较。

### Arm B Stage 1/2 实际运行结果（2026-09-16）

#### Arm B Stage 1：MM-WLAuslan 姿态适配

- 运行目录：`MyDrive/auslan_work/runs/arm_bc__csl_stage1_weight__b1_pose/`。
- 初始化：`csl_stage1_weight.pth`；使用完整 Uni-Sign pose-only 模型。
- 数据：MM-WLAuslan `Train/studio`，38,580 段；该阶段没有连续句子验证集。
- 参数：batch size 16，gradient accumulation 1，有效 batch size 16；每 epoch 2,411 个 optimizer step；3 个 epoch，共 7,233 步；`num_workers=8`。
- 优化器：AdamW，基础 `lr=1e-4`，`weight_decay=0.01`，梯度裁剪 1.0；warmup 5%，之后 cosine decay。
- 分组学习率：pose encoder 为 `1e-4`，temporal 和 decoder 各为 `1e-5`；三组都更新，但 temporal/decoder 只用基础学习率的 0.1 倍。
- 结果：已正常完成，`checkpoint.pt` 已生成。B1 的交付物是姿态适配后的 checkpoint，不是连续翻译 BLEU 分数；训练耗时约 35 分钟。

#### Arm B Stage 2：Auslan-Daily 连续句微调

- 运行目录：`MyDrive/auslan_work/runs/arm_bc__csl_stage1_weight__b2_daily/`。
- 初始化：B1 的 `checkpoint.pt`，不是重新从 CSL checkpoint 开始。
- 数据：Auslan-Daily train 21,998 段、val 1,492 段；使用 131 段 AD gate 排除名单，其中 train/val 实际排除 119 段。
- 参数：全部参数解冻（约 587.75M/587.75M）；batch size 8，gradient accumulation 1，有效 batch size 8；每 epoch 2,749 步；20 个 epoch，共 54,980 步；`num_workers=8`。
- 优化器：AdamW，`lr=3e-5`，`weight_decay=0.01`，梯度裁剪 1.0；warmup 5%，之后 cosine decay。验证生成使用 5 beams、`max_new_tokens=100`、`label_smoothing=0.2`。
- loss 走势：epoch 0 平均约 8.50，epoch 7 约 5.98，epoch 14 约 5.68，epoch 19 约 5.75。大约从第 4～5 个 epoch 开始进入平台期，后期继续下降非常有限。训练耗时约 4.5 小时。

Stage 2 验证集结果如下。Communication 和 News 分开报告，不能取平均：

| 子集 | 解码 | BLEU-1 | BLEU-4 | ROUGE-L | 不同输出 | looping | hyp/ref 长度 |
|---|---|---:|---:|---:|---:|---:|---:|
| Communication | plain | 26.33 | 3.40 | 12.49 | 9.1% | 2.1% | 4.9 / 5.2 |
| Communication | nr3 | 26.28 | 3.43 | 12.53 | 8.7% | 0.0% | 4.7 / 5.2 |
| Communication | nr3 + repetition penalty 1.2 | 26.68 | 3.53 | 12.86 | 10.1% | 0.0% | 4.6 / 5.2 |
| News | plain | 6.86 | 0.54 | 6.23 | 38.3% | 66.6% | 29.0 / 15.9 |
| News | nr3 | 18.25 | 1.21 | 9.62 | 34.3% | 0.0% | 16.2 / 15.9 |
| News | nr3 + repetition penalty 1.2 | 18.25 | 1.18 | 9.34 | 35.7% | 0.0% | 16.8 / 15.9 |

结果判断：

- News 的 plain 解码受到严重重复生成影响：平均输出长度 29.0，而参考长度只有 15.9，66.6% 的输出出现循环。加入 `no_repeat_ngram_size=3` 后 looping 降到 0，BLEU-4 从 0.54 提升到 1.21；因此 plain 的 News 分数不能代表模型的全部能力。
- Communication 的三种解码结果几乎不变，最佳 BLEU-4 只有 3.53，且不同输出占比约 10%，说明模型仍有明显的低多样性、欠拟合或模式坍缩问题。解码约束没有解决主要问题。
- 与当前官方 Arm A Stage3 的 `nr3` 参考相比，B2 最好结果仍明显偏低：Communication 为 BLEU-4 3.53 / ROUGE-L 12.86，对比官方 11.69 / 24.01；News 为 1.21 / 9.62，对比官方 2.63 / 13.35。
- B2 和官方 Stage3 不是只差数据策略：官方使用 `lr=3e-4`、有效 batch size 32、`weight_decay=1e-4`、无 warmup；B2 使用 `lr=3e-5`、有效 batch size 8、`weight_decay=0.01`、5% warmup。因此不能把差距全部归因于 Arm B 的数据顺序。
- 当前 B2 应保留为一个有效的对照/诊断结果，不建议用相同配置重复训练。若要救援 B2，应新建不同 run，优先改变学习率和优化策略，而不是对当前已完成 checkpoint 原样续训。

第 12 节已完成 B2 的 `nr3` 和 `nr3_rp12` 评估；第 13 节已完成比较。Arm C 的 `aux_weight=0.1/0.2/0.3` 当时均未完成，因此被自动跳过，尚无 C 结果。

### 起点 checkpoint 对比实验（OpenASL 已跑完，How2Sign 尚未运行；2026-09-17 更新）

目的：检验"解码器已经会输出英文的起点，是否比 CSL 起点更适合这个任务"。只改初始化一个变量，其余与已完成的官方 Stage3 run 完全相同，因此分数差异可以归因到起点。B2 的结果明显低于官方 Stage3，说明换数据策略没有立刻见效，换起点是目前更值得先试的一个杠杆。

- `colab_train_how2sign_init.ipynb` → `how2sign_pose_only_slt.pth`，run 名 `arm_a__how2sign_pose_only_slt__official_stage3__single_a100__bf16`
- `colab_train_openasl_init.ipynb` → `openasl_pose_only_slt.pth`，run 名 `arm_a__openasl_pose_only_slt__official_stage3__single_a100__bf16`

两份 notebook 由同一个生成脚本产出，逐字比对过，只差 checkpoint 名、sha256、run 名和说明文字。

**参数核对（逐项对到出处，2026-09-16）**：配置里的每一项都和 Uni-Sign 官方 Stage 3 一致。核对来源是本地仓库 `Uni-Sign/`（commit `eed438b`）的实际代码默认值，以及论文 Table 2。

| 项目 | 我们的配置 | 官方 | 出处 |
|---|---|---|---|
| 优化器 | AdamW | AdamW | `script/train_stage3.sh --opt AdamW` |
| 学习率 | 3e-4 | 3e-4 | `script/train_stage3.sh --lr 3e-4` |
| weight decay | 1e-4 | 1e-4 | `utils.py` 的 `--weight-decay` 默认 0.0001 |
| betas | (0.9, 0.999) | (0.9, 0.999) | `utils.py` 的 `--opt-betas` 默认 None，timm 走 torch AdamW 默认值 |
| eps | 1e-9 | 1e-9 | `utils.py` 的 `--opt-eps` 默认 1.0e-09 |
| 梯度裁剪 | 1.0 | 1.0 | `utils.py` 的 `--gradient-clipping` 默认 1.0，写进 deepspeed config |
| 调度 | cosine，无 warmup，衰减到 0 | 同 | `fine_tuning.py` 的 `get_scheduler(name='cosine')`；`--warmup-epochs` 默认 0 |
| epoch | 20 | 20 | `script/train_stage3.sh`；论文 Table 2 |
| 有效 batch | 8 × 累积 4 = 32（单卡） | 8 × 4 卡 × 累积 1 = 32 | `script/train_stage3.sh`；论文 Table 2 的 "batch size: 8, gradient accumulation: 1" |
| label smoothing | 0.2 | 0.2 | `utils.py` 的 `--label_smoothing` 默认 0.2，`models.py` 传给 CrossEntropyLoss |
| max_length | 256 | 256 | `utils.py` 的 `--max_length` 默认 256 |
| 生成 | num_beams=4、max_new_tokens=100 | 同 | `fine_tuning.py` 第 264–265 行 |

论文 Table 2 只给了统一的一套 Stage 3 设置，没有按下游数据集分别列参数，也没有写 How2Sign / OpenASL 是从 stage1 还是 stage2 权重开始微调。所以"和论文参数一致"这句话的准确含义是：和 Uni-Sign 唯一的那套 Stage 3 设置一致，不存在一套单独的 How2Sign 参数或 OpenASL 参数。

**与官方仍然存在的差异（必须在汇报里写明）**：

1. **起点性质不同**。官方 Stage 3 把这套参数用在**预训练**权重上；这两个 run 用在**已经针对 SLT 微调过**的权重上。3e-4 有可能冲掉那个英文解码器学到的东西。保留 3e-4 是为了只动一个变量；如果结果不如基线，下一步是从同一 checkpoint 单独跑一个 `lr=1e-4` 的 run，用新 run 名，当作另一个实验。
2. **精度已对齐，但基线没有**。`train.py` 现在支持 `precision: bf16`（CUDA autocast，参数和 AdamW 状态仍是 fp32，验证解码仍是 fp32），这两个 run 都用 bf16，和官方的 DeepSpeed bf16 一致。**但已完成的 CSL 基线是 fp32 跑的**，所以和它比是双变量。两个新 run 之间（How2Sign vs OpenASL）仍然是单变量，干净。要让对 CSL 的比较也单变量，需要跑 notebook 第 10 节的 bf16 CSL 对照（约 3 小时），见下。
3. **单卡的 BatchNorm 统计口径不同**。官方 4 卡会做 SyncBatchNorm，BN 统计跨 32 个样本；我们单卡累积，BN 只在 8 个样本上统计。梯度的有效 batch 相同，BN 不同。
4. **最终权重的选法不同**。官方每轮在 dev 上评估并保存 `best_checkpoint.pth`；我们的 `train.py` 只在训练结束后用最后一轮权重评估一次。这一点和已完成的官方 Stage3 run 相同，所以三个起点之间仍然可比。

**notebook 里比 `colab_train_official.ipynb` 多加的两个 gate**：

1. 姿态完整性只检查 `auslandaily` 行。`manifest.jsonl` 现在是联合 manifest，还包含 5.1 万段 MM-WLAuslan；Arm A 用不到，不该拦住这个 run。
2. 排除名单加了硬 gate：统计 `ad-` 开头的 uid，少于 131 就直接停下。原因见第 7 节——MM-WLAuslan 那次检查会往同一个 `excluded.txt` 写，可能已经把 Auslan-Daily 的名单清空；那样训练数据和基线就不是同一套，分数不可比。

**已核实**：三个 checkpoint 都是 `{'model': ...}`、627 个张量、形状逐一相同，所以是即插即用，`missing=0 / unexpected=0` 有保证，不需要改 `model_adapter.py`。sha256 已写进 notebook：`how2sign_pose_only_slt.pth` 是 `1bfd5f3312f04e4736f0a52f4ef9535916e6de9676a2a0d00c708748683fb00d`，`openasl_pose_only_slt.pth` 是 `f836ea66bc837bbe6ed717a4b9bece87875f03ef96d4bf1092ca3dd767982798`。两个文件在固定 revision `eab251b7` 的 HF 仓库根目录都存在。

**训练拆成两段（2026-09-16）**：每个 run 先只跑 epoch 0（约 14 分钟），看完第 1 轮 loss 再决定是否付剩下的 19 个 epoch（约 4.3 小时）。这两个起点已经针对 SLT 微调过，3e-4 且无 warmup 有可能在第 1 轮就把英文解码器冲掉；epoch 0 的平均 loss 和重跑的 Arm A 基线 6.707 相比，就是这个问题最便宜的信号。

- 明显低于 6.707（约 6.41 以下）：英文解码器保留下来了，继续跑 1–19。
- 接近或高于 6.707（约 6.61 以上）：大概率已经被冲掉，继续跑四小时很可能只是复现基线；更值得做的是从同一 checkpoint 另起一个 `lr=1e-4` 的 run。
- notebook 第 6 节跑 epoch 0，第 6b 节打印对比和判断，第 6c 节是可选的 lr/batch 搜索，第 6d 节续训到结束。第 6d 节有闸门 `CONTINUE_TRAINING = False`，所以「全部运行」会在 epoch 0 之后停住，不会直接烧掉几个小时。

为此给 `train.py` 加了 `--stop-after-epochs N`：在 epoch 边界干净停下、存 checkpoint、退出码 130，之后用 `--resume` 继续。**它不改变训练内容**：cosine 调度的 `total_steps` 仍然来自 `optim.epochs`（20 轮），而且这个开关不进续训指纹，所以续训时不带它也是同一个 run。`test_resume.py` 新增了 `paused` 场景，Arm A、B1、C 三个配置下"暂停再续训"的最终权重都和不中断训练逐位相同。

**精度与 bf16 对照（2026-09-16）**：两个 notebook 的配置都写了 `precision: bf16`，run 名带 `__bf16` 后缀，和 fp32 的 CSL 基线目录区分开。

- 预计耗时：epoch 0 约 8–10 分钟，其余 19 个 epoch 约 2.5–3 小时（fp32 时分别是 14 分钟和 4.3 小时）。以日志里的实际秒/步为准。
- notebook 第 10 节是可选的 **bf16 CSL 对照**（`arm_a__csl_stage1_weight__official_stage3__single_a100__bf16`，默认 `RUN_CONTROL = False`）。两个 notebook 里哪个先跑都行，跑一次即可，另一个的对比表会自动读到。在引用"某个起点比 CSL 高 N 分"之前，应该先有这个对照。
- 第 6b 节的 epoch 0 判据用的 6.707 来自 2026-09-16 重跑的 Arm A 官方 Stage3 run，该 run 用的精度待确认。精度对 loss 的影响通常远小于 0.1，相对 0.3 的判断带不算大，但在确认之前这不算严格同口径的对比，只能当便宜的信号。
- 已实测：`precision` 进续训指纹，fp32 和 bf16 的指纹不同，所以一个 run 不会被用另一种精度悄悄续上。**注意一个边角情况**：配置里显式写 `precision: fp32` 和完全不写这个键，指纹也不同。已经在跑、配置里没有这个键的 run（比如 `colab_train_bc.ipynb` 的 Arm C），不要事后往配置里补 `precision: fp32`，否则 `--resume` 会被指纹守卫拒掉。
- 改完 `train.py` 后重跑了 `test_resume.py`：Arm A/B1/C × 三种中断 + 三项守卫，12 项全部 PASS（fp32 路径，CPU）。bf16 路径需要 CUDA，本地测不了，Colab 上第一次跑时看日志里的 `precision=bf16` 确认。

**可选的 lr / batch 搜索（notebook 第 6c 节，默认 `SEARCH = False`）**：不是完整网格，是围绕论文那一点的十字搜索，5 个代理 run，每个 2 个 epoch、验证集只评 400 段，合计约 2–2.5 小时。完整 3×3 网格要 9 个 run、约 4 小时，按当前 GPU 预算不值得。

- 扫的点：`lr` 取 3e-4 / 1e-4 / 3e-5（有效 batch 固定 32），有效 batch 取 16 / 32 / 64（`lr` 固定 3e-4）。
- **micro-batch 始终是 8**，只改 `gradient_accumulation_steps`。这样有效 batch 变了，而 BatchNorm 的统计口径不变，避免一次动两个东西。
- 代理 run 写在 `auslan_work/runs/search/` 下，名字形如 `search__<ckpt stem>__lr3e-04__eb32__bf16`，各自独立、可续训、已完成的会跳过。
- **三条限制**：①2 个 epoch 的代理有自己的 cosine 调度，在两轮内就衰减到 0，它给的是短预算下的排序，不能证明这个排序在 20 个 epoch 下仍然成立；②lr 和 batch 相互影响，如果赢的 lr 不是 3e-4，值得再补一个"赢的 lr × 赢的 batch"的点；③在验证集上挑参数又在验证集上报分数等于在自己的指标上调参，赢的配置必须换新 run 名重跑完整训练，最终成绩要在 test 集上、解码设置事先固定。
- **搜出来的配置就不再是官方 Stage3 recipe 了**，要作为一个单独的 tuned run 和官方那个并列汇报，不能当作对官方 run 的修正。

**OpenASL 起点的 epoch 0 结果（2026-09-17，完整结果见下一小节）**：`arm_a__openasl_pose_only_slt__official_stage3__single_a100__bf16`。

- 加载检查全部通过：`precision=bf16`、`missing=0 unexpected=0`、587.75M/587.75M 可训练、train 21,998 / val 1,492、排除 119 段、`effective_batch=32`、`updates/epoch=687`。
- epoch 0 已记录的 loss 从 5.72 降到约 5.40，平均约 **5.49**，比 Arm A 基线的 6.707 低约 **1.22**，远超判据的 0.3 带。**结论：OpenASL 的英文解码器在 3e-4 下保留下来了，不需要跑 6c 的 lr/batch 搜索，直接继续 1–19。**
- 值得注意的是第一个记录点（step 50）就已经是 5.72，本身就低于基线整个 epoch 0 的均值，说明这不只是"没被冲垮"，而是起点从一开始就在贡献。
- transformers 关于 `shared.weight` 和 `lm_head.weight` 未绑定的警告是预期的：这个 checkpoint 存了一个独立的、已经微调过的 lm_head，不绑定才能保留它。不是错误。
- 速度：约 0.87 秒/optimizer step，一个 epoch 约 10 分钟，其余 19 个 epoch 约 3.2 小时（比原先估的 2.5–3 小时略多）。checkpoint 约 7 GB，写 Drive 约 22 秒。
- **提醒**：loss 低不等于 BLEU 高，最终仍以 Communication / News 分开的验证集指标为准。另外 How2Sign 起点必须用同一套配置跑，否则两个起点之间不是单变量对比。

**判读方式**：和官方 Stage3 run 的 plain 行比（Communication BLEU-4 11.51，News 2.17），差距要超过噪声底线（Communication 0.63，News 0.22）才算有效。建议先跑 OpenASL：它的 ASL 训练数据约 288 小时、多手语者，比 How2Sign 的约 80 小时单人棚拍更接近 Auslan-Daily 的实拍多人场景。每个 run 在 A100 上约 4.5 小时。notebook 第 9 节会把三个起点放在同一张表里比较。

### OpenASL 起点完整结果（2026-09-17，已完成：新的最佳 Arm A）

run：`arm_a__openasl_pose_only_slt__official_stage3__single_a100__bf16`，notebook `colab_train_openasl_init.ipynb`。

- 环境：A100-SXM4-40GB，`torch 2.11.0+cu128`，`transformers 5.16.1`。checkpoint sha256 校验通过，Uni-Sign 代码 `eed438b`。
- 数据 gate：联合 manifest 76,549 段、Auslan-Daily 25,109 段、姿态缺失 0、AD 排除名单 131 段（train/val 实际排除 119 段），train 21,998 / val 1,492。和 CSL 官方 run 是同一套数据。
- 配置：与官方 Stage3 完全一致（lr 3e-4、wd 1e-4、betas (0.9, 0.999)、eps 1e-9、clip 1.0、cosine 无 warmup、20 epoch、8×4=32、`label_smoothing=0.2`、`max_length=256`、4 beams、`max_new_tokens=100`），`precision=bf16`。
- 训练流程：epoch 0 单独跑完暂停，然后从 `ckpt_step000000687.pt` 续训 1–19，中间没有再断线。13,740 步全部完成，`checkpoint.pt` 已保存。
- 速度：续训的 13,053 步用时约 11,700 秒（约 0.9 秒/步，含每轮存 7 GB checkpoint 的约 22 秒），加上 epoch 0 约 10 分钟，训练本身合计约 **3.4 小时**（不含最后的验证解码）。

每个 epoch 的平均训练 loss（每 50 步记一个点）：

| epoch | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| loss | 5.48 | 5.13 | 4.82 | 4.58 | 4.48 | 4.25 | 4.06 | 3.93 | 3.84 | 3.70 |

| epoch | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| loss | 3.67 | 3.62 | 3.51 | 3.45 | 3.43 | 3.42 | 3.40 | 3.40 | 3.37 | 3.36 |

验证集结果（plain = `num_beams=4`，不加解码技巧，这是用于对比的行）：

| 子集 | 解码 | n | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | ROUGE-L | 不同输出 | looping | hyp/ref 长度 | OOV type |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Communication | plain | 792 | 43.99 | 29.12 | 22.41 | **18.03** | **31.67** | 71.7% | 0.6% | 4.9 / 5.2 | 0.8% |
| Communication | nr3 | 792 | 43.96 | 29.12 | 22.42 | 18.06 | 31.69 | 71.7% | 0.0% | 4.9 / 5.2 | 0.8% |
| News | plain | 700 | 26.01 | 12.54 | 7.61 | **5.20** | **18.86** | 97.3% | 12.9% | 15.9 / 15.9 | 10.2% |
| News | nr3 | 700 | 26.16 | 12.65 | 7.73 | 5.28 | 19.09 | 97.3% | 0.0% | 15.6 / 15.9 | 10.2% |

和 CSL 起点对比（plain 行）：

| 子集 | 指标 | CSL fp32 | CSL bf16 | OpenASL bf16 | vs CSL bf16（单变量） | vs CSL fp32 |
|---|---|---:|---:|---:|---:|---:|
| Communication | BLEU-1 | 36.12 | 33.62 | 43.99 | +10.37 | +7.87 |
| Communication | BLEU-4 | 11.51 | 9.77 | 18.03 | **+8.26** | **+6.52** |
| Communication | ROUGE-L | 23.78 | 20.55 | 31.67 | +11.12 | +7.89 |
| Communication | 不同输出 | – | 40.9% | 71.7% | +30.8 pt | – |
| News | BLEU-1 | 17.39 | 16.85 | 26.01 | +9.16 | +8.62 |
| News | BLEU-4 | 2.17 | 1.82 | 5.20 | **+3.38** | **+3.03** |
| News | ROUGE-L | 12.90 | 12.33 | 18.86 | +6.53 | +5.96 |
| News | looping | 34.6% | 37.9% | 12.9% | -25.0 pt | -21.7 pt |

结果判断：

- **换起点是目前为止最大的一个杠杆。** BF16 CSL 对照已经在 `colab_train_official.ipynb` 里跑过，所以“OpenASL bf16 vs CSL bf16”是严格的单变量对比：Communication BLEU-4 +8.26（约为噪声底线 0.63 的 13 倍），News +3.38（约为 0.22 的 15 倍），几乎翻倍到接近三倍。即使和更强的 FP32 CSL 比（双变量），差距仍有 +6.52 / +3.03。**OpenASL bf16 取代 CSL fp32，成为当前最佳 Arm A。**
- **提升不只是 n-gram 分数。** Communication 不同输出占比从 40.9% 升到 71.7%，说明高频句坍缩明显减轻，模型更多是在按姿态区分句子；News 的平均输出长度 15.9 和参考完全一致，looping 从 35%～38% 降到 12.9%。
- **`nr3` 已经基本没用了。** Communication +0.03、News +0.08，都远在噪声底线内。CSL 起点时 News 靠 `nr3` 涨 0.46，现在重复问题主要由更好的解码器自己解决了。最终 test 评估可以直接固定用 plain，不用再在验证集上挑解码设置。
- **训练 loss 与判据一致。** epoch 0 就比 CSL 基线低 1.23，最后一轮 3.36，而 CSL fp32 最后约 4.51；离 label smoothing 0.2 的理论下界估算 2.99 只差约 0.37。最后 5 个 epoch 只从 3.43 降到 3.36，学习率已衰减到 0，已经收敛到这套调度的终点。训练 loss 这么低还没有看到验证分数变差，但因为只评估了最后一轮，无法判断中间某一轮是否更好。
- **3e-4 没有冲掉英文解码器。** 之前担心的“已经 SLT 微调过的权重用 3e-4、无 warmup 会被冲掉”没有发生，所以原计划里的 `lr=1e-4` 对照不再是必需的；如果要做，只能作为调参实验，不能当作补救。
- **News 仍是瓶颈，而且剩下的主要是词汇问题。** OOV type 仍是 10.2%（290/2831，token 2.7%），未见词大多是人名、地名和新闻词（kambosos、trafford、gillard、jinping's、separatists 等），这是训练数据的属性，换起点解决不了。News 的 BLEU-4 5.20 仍然只有 Communication 的 29%。

必须写明的限制：

1. **单 seed。** 噪声底线 0.63 / 0.22 是在旧的 3e-5 CSL 配置上用两个 seed 测出来的，不是这套配置的噪声。差距远大于这个底线，结论方向基本不会翻，但最终汇报要写明是单 seed。
2. **全部是验证集分数，而且没有 test 结果。** 最终成绩要在完整 test 集上报（不套用 `excluded.txt`），解码设置事先固定为 plain。
3. **OpenASL 起点多用了约 288 小时 ASL 翻译数据的监督**，所以这个提升等于“多了一次英文 SLT 预训练”，不是纯粹的算法改进；汇报时要说明起点不同。许可证仍是 CC BY-NC 4.0。
4. How2Sign 起点还没跑，所以还不能说“OpenASL 是最好的英文起点”，只能说“英文 SLT 起点明显优于 CSL 起点”。
5. transformers 关于 `shared.weight` / `lm_head.weight` 不绑定的警告在训练和 `--eval-only` 时都出现，是预期行为（见 epoch 0 小节）。SacreBLEU 的 tokenized period 警告同之前，各 run 使用同一评估脚本，相对比较有效。

### 当前进行中 / 下一步（2026-09-17）
- **近期不做 official seed 1**：`colab_train_official.ipynb` 的 seed 0 已经完成并作为当前最佳 Arm A 参考。`colab_train.ipynb` 里的 `LR=1e-4` 仍只是可选的控制变量消融，不是当前主线。
- **Arm B/C 当前状态**：MM-WLAuslan 的正面 `kf` 姿态已经抽取完成；Arm B Stage 1 和 Stage 2 已完成，实际参数与结果见上面的“Arm B Stage 1/2 实际运行结果”。Arm C 的三个辅助权重目前还没有完成。使用 `colab_train_bc.ipynb` 时，真实 Uni-Sign、联合 `manifest.jsonl`、Drive 上的姿态 tar 分片和断点续训机制均已配置好；断线后重跑准备 cell 和对应训练 cell 即可续训。
  1. **Arm B stage 1**：只读 MM-WLAuslan `Train` 的 `studio` 子集（38,580 段），训练 pose encoder；temporal 和 decoder 按配置使用 1/10 学习率。默认 `lr=1e-4`、batch 16、3 epoch；此阶段没有连续句子验证集，不把 gloss 生成分数当最终结果。
  2. **Arm B stage 2**：必须从 B1 的 `checkpoint.pt` 初始化，改在 Auslan-Daily train 上全部解冻微调，并在 Auslan-Daily val 上评估。顺序不能反。
  3. **Arm C**：从 `csl_stage1_weight.pth` 单阶段开始，Auslan-Daily 是主损失，MM-WLAuslan 是辅助损失；分别跑 `aux_weight=0.1/0.2/0.3`，不能从 B1 checkpoint 开始。
- **开始 B/C 前的 gate**：联合 manifest 必须同时包含 Auslan-Daily 和 MM-WLAuslan；姿态缺失数为 0；`excluded.txt` 必须仍是 Auslan-Daily 的 131 段异常 UID（B2/C 的 train+val 实际排除 119 段），不能使用 MM-WLAuslan 检查生成的空名单。B1 的 MM-WLAuslan 没有需要排除的片段。
- **B/C 的 run 只和同一解码设置比较**：B2/C 完成后评估 `plain`、`nr3`、`nr3_rp12`，Communication 和 News 分开报告，不取平均。最终 test 评估不能套用 `excluded.txt`，并且要事先固定解码设置。
- **解码修正**：
  - 配置里有独立的 `decode` 项，不进续训指纹。
  - `--eval-only` 用 run 的 `checkpoint.pt` 重新评估，结果写进 `eval_<tag>/`，不动原有结果。
  - 笔记本比较两种设置：`nr3`（`no_repeat_ngram_size=3`）和 `nr3_rp12`（再加 `repetition_penalty=1.2`）。
  - 注意：在验证集上挑解码设置，又在验证集上报分数，相当于轻微地在验证集上调参。最终在 test 集上报成绩时，要事先固定一种设置。
- `evaluate.py` 新增了几项指标：`unique_hyps`（不同输出的占比）、`looping`（含重复词三元组的输出占比）、`hyp_len` / `ref_len`（输出和参考译文的平均长度）。
- **OpenASL 起点已跑完，是新的最佳 Arm A**（Communication BLEU-4 18.03、News 5.20，plain），见上面的“OpenASL 起点完整结果”。How2Sign 起点（`colab_train_how2sign_init.ipynb`）还没跑。
- **新版 Arm B notebook 已写好，尚未运行（2026-09-17）**：`colab_train_b_openasl.ipynb`。用户因 GPU 紧张决定**不跑 OpenASL seed 1，直接跑新版 Arm B**；Arm C 暂不做。
  - 起点 `openasl_pose_only_slt.pth`。run 名 `arm_b__openasl_pose_only_slt__b1_pose_mt5frozen__bf16` 和 `arm_b__openasl_pose_only_slt__b2_official_stage3__single_a100__bf16`，不覆盖旧的 CSL B1/B2。
  - **B1**：MM-WLAuslan `Train/studio` 38,580 段；**整个 mT5（decoder 组，582M）冻结**，pose encoder `1e-4`、temporal `1e-5`；其余沿用旧 B1（wd 0.01、batch 16、3 epoch、5% warmup、cosine），bf16。冻结解码器是为了保护 OpenASL 起点已经会输出英文句子的解码器，避免被单词 gloss 目标带偏。约 30–40 分钟。
  - **B1 目标改为小写（2026-09-17）**：第一次 B1（`..._b1_pose_mt5frozen__bf16`）目标是大写 gloss（`WHALE`），loss 在前 1.8 个 epoch 只从约 9.8 降到 7～7.5。判断：解码器冻结后学不会输出大写，loss 被抬高，而且 encoder 被推向解码器做不到的输出格式。该 run 已手动停止、不使用。现在 B1 用 `gloss_text_mode: strip_paren_lower`，run 名改为 `arm_b__openasl_pose_only_slt__b1_pose_mt5frozen_lower__bf16`。B2 不受影响（Auslan-Daily 用原句）。即使改成小写，B1 loss 也不会低到 Arm A 的 3.x：3,215 类、每类约 12 个样本、只有 5.35M 可训练参数，另有 label smoothing 约 2.99 的下界。B1 的好坏以 B2 的 epoch 0 闸门为准。实测速度约 0.19 秒/步，全程约 23 分钟；checkpoint 约 2.4 GB。
  - **B2**：从 B1 的 `checkpoint.pt` `--init-from`，优化器和 OpenASL Arm A 逐项相同（官方 Stage3、bf16、4 beams、seed 0）。如果 Drive 上有 Arm A 的配置文件，notebook 会逐项比对，不一致就停。所以 **B2 vs OpenASL Arm A 只差“是否先跑 B1”一个变量**。
  - **epoch 0 自动闸门**：B2 先跑 epoch 0（约 10 分钟），平均 loss 与 OpenASL Arm A 的 5.48 比较；高出超过 0.3 就不自动续训（可用 `FORCE_CONTINUE` 覆盖）。
  - 数据 gate：检查联合 manifest（AD 25,109、MMWL train/studio 38,580）、姿态无缺失、`excluded.txt` 恰好 131 个 `ad-` uid，否则尝试从 `excluded_auslan_daily.txt` 修复。
  - 本地已用 smoke 后端验证：B1 后 decoder 0/40 张量变化、pose_encoder 16/16 与 temporal 27/27 变化；B2 “epoch 0 暂停再带 `--init-from` 续训”与不中断训练逐位相同。bf16 路径只能在 Colab 上确认。
  - 只评 plain（OpenASL Arm A 上 nr3 已无作用）；最后一节把 CSL A、旧 CSL B2、OpenASL A、OpenASL B 放在一张表里，并打印 B−A 的差值和噪声底线判定。
- **对 B/C 的影响**：已完成的 B1/B2 和计划中的 Arm C 都是从 `csl_stage1_weight.pth` 开始的，而 CSL 起点比 OpenASL 低一大截，所以它们现在的比较对象应该是 OpenASL Arm A。B2 最好成绩（Communication 3.53 / News 1.21）已经远落后。**建议**：Arm C 不要再用 CSL 起点跑三个 `aux_weight`；如果还要验证 MM-WLAuslan 是否有帮助，应改成从 `openasl_pose_only_slt.pth` 开始、使用同一套官方 Stage3 + bf16 配置，run 名用新的，和 OpenASL Arm A 做单变量对比。
- 之后的候选，建议按优先级：
  1. OpenASL 起点补一个 seed 1，测这套配置本身的噪声底线（之前的 0.63 / 0.22 来自旧配置）。
  2. 在 OpenASL 起点上做 B/C 类的 MM-WLAuslan 实验（见上）。
  3. How2Sign 起点：用来确认“哪个英文起点更好”，但预计差异比 CSL vs 英文起点小得多，优先级低于 1、2。
  4. News 的 OOV（10.2%）是剩下的主要瓶颈，可重新评估 Arm D 或专有名词相关的方法。
  5. 在上面这些方向定下来之后，再用完整 test 集（不套用排除名单、固定 plain 解码）报最终成绩。
- `lr=1e-4` 对照原本是“换起点不如基线时”的补救方案，现在不再需要。

## 4. 已核实的事实（原计划里有错的地方，以这里为准）

- **不是四个互不共享的编码器**：`models.py` 里左右手共享同一个编码器，四个分支只有三套权重。
- **提取器不是 RTMPose-x / MMPose**：Uni-Sign 自己用的是 `rtmlib.Wholebody(mode="lightweight")`，即 RTMW-l-m（192×256）+ YOLOX-tiny。我们与它完全一致。
- **身体也做了归一化**：`crop_scale` 用整段视频的一个包围框把身体映射到 [-1,1]。手和脸先减各自的 root，再除以同一个身体尺度。
- **脸部 18 点的顺序**：下颌 9 点 + 内唇 8 点 + **鼻尖在最后**（鼻尖是图的中心，也是 root）。
- `spec.py` 移植的归一化和 Uni-Sign 仓库的实现逐数值比对过，最大误差 3e-08。
- **选人**：画面里有多个人时，每帧选检测框最大的那个（`largest`）。用 Auslan-Daily 官方姿态标注做了验证：
  - 留出的 49 段上，逐帧正确率 largest 99.97%，track 97.99%，first 98.75%。
  - 开发集 38 段上，largest 99.96%。
- **批量推理**：`--pose-batch 32` 和逐帧推理对比过，14 个用例全部一致（`verify_batching.py`）。
- **断点续训**：`test_resume.py` 覆盖 Arm A、B1、C，分别测"被杀掉"、"按 Stop"和"用 `--stop-after-epochs` 主动暂停"三种中断，续训后最终模型和不中断的结果逐位相同（2026-09-16 实测全部 PASS）。检查点每 30 分钟存一次，保留最新 2 份，每份约 7 GB。
- **脸部支路的参数**（用 `csl_stage1_weight.pth` 实测）：85 个张量，152 万参数。形状和节点数有关的只有 6 个可学习邻接矩阵（每个 2×18×18），卷积权重和节点数无关，节点特征最后取平均。整个脸部支路只占全模型 5.88 亿参数的 0.26%。
- **指纹**：`spec.SCHEMA_FINGERPRINT` 覆盖 spec 版本、各部位选哪些点、root、置信度阈值。**改选点会被当成"要重新抽取"而报错**，虽然 133 个点其实都存在 `.npz` 里。反过来，**归一化的计算代码不在指纹里。**

## 5. 代码地图（`unisign/`）

| 文件 | 作用 |
|---|---|
| `spec.py` | 唯一真相源：选点、归一化（`load_part_kp`）、指纹。`FACE_UNMODELLED` 记录脸部哪些通道没被建模 |
| `manifest.py` / `auslan_daily.py` | 生成 `manifest.jsonl`，每行一段：uid、dataset、subset、split、text、视频路径。视频可以直接用 `archive.zip::member` 的形式在压缩包里读，不用解压 |
| `extract_pose.py` | 用 rtmlib 抽 133 点，存成 `.npz`（keypoints、scores、meta、n_people、chosen_box）。支持分片、批量推理、跳过已抽取的，spec 或策略变了会重新抽 |
| `verify_pose.py` | 训练前检查：一致性、质量统计、镜像检测，写 `--exclude-out` 排除名单，画 `--overlay-flag` 叠加视频 |
| `verify_batching.py` | 验证批量推理和逐帧推理结果一致 |
| `dataset.py` | 数据集和 collate，移植自 Uni-Sign：用最后一帧补齐，长片段做时序子采样。支持 `exclude` |
| `train.py` | 训练 Arm A/B/C。有 smoke 和 unisign 两个后端，支持冻结和分组学习率、断点续训、`data.exclude`、`--set` 覆盖配置、`decode` 解码选项、`--eval-only` 只评估、`--stop-after-epochs N` 在 epoch 边界暂停（不改调度、不进指纹）、`precision: fp32|bf16`（进指纹） |
| `test_decode.py` | 测解码选项和 `--eval-only`：无选项时和 Uni-Sign 自己的 generate 调用一致；只评估模式不动原有结果；改 `decode` 不影响续训 |
| `model_adapter.py` | 加载 Uni-Sign（`models.Uni_Sign(args=...)`），参数分组：pose_encoder / temporal / decoder |
| `evaluate.py` | 算 BLEU（sacrebleu）、ROUGE-L、OOV，按子集分组，没有合并均值 |
| `stitch.py` | Arm D：把孤立词拼成伪句子 |
| `smoke_model.py` | 小模型，只用来测试整条流程能不能跑通，它的分数不是结果 |
| `test_resume.py` | 断点续训测试 |
| `configs/arm_a.yaml` 等 | 各个 Arm 的配置。默认后端是 smoke；Colab 训练笔记本会自动换成 unisign |
| `colab_setup.ipynb` | Colab：装依赖 → 生成 manifest → 把压缩包暂存到本地盘 → 多进程抽取（14 个 worker）→ 训练前检查 |
| `colab_train.ipynb` | Colab：下载模型 → 解压姿态 → 生成配置 → 用不同解码设置重评已完成的 run → 训练 `LR`/`SEEDS` 指定的 run → 对比表 |
| `colab_train_official.ipynb` | Colab：按官方 Stage3 优化器配置训练单卡版本，使用梯度累积模拟有效 batch size，并评估 plain / `nr3` 解码 |
| `colab_train_how2sign_init.ipynb` | Colab：官方 Stage3 参数 + `precision: bf16`，只把初始化换成 `how2sign_pose_only_slt.pth`；训练拆成 epoch 0 / 1–19，末尾对比各 run，并可选跑 bf16 CSL 对照 |
| `colab_train_openasl_init.ipynb` | Colab：同上，初始化换成 `openasl_pose_only_slt.pth` |
| `colab_train_b_openasl.ipynb` | Colab：OpenASL 起点的新版 Arm B。B1 冻结 mT5 → B2 按 OpenASL Arm A 的官方 Stage3 配置微调 → epoch 0 自动闸门 → 与 Arm A 对比 |
| `colab_train_bc.ipynb` | Colab：恢复联合姿态 → 检查 B/C 的数据和排除名单 → 顺序训练 Arm B stage 1/stage 2 → 扫描 Arm C 的 `aux_weight` → 评估并比较各 run |

本地资源（项目根目录）：
- `Uni-Sign/`：GitHub 原版，commit `eed438b`，没改过。`pretrained_weight/mt5-base` 是 `google/mt5-base`。
- `checkpoints/`：`csl_stage1_weight.pth`、`how2sign_pose_only_slt.pth`、`openasl_pose_only_slt.pth`，来自 HuggingFace 的 `ZechengLi19/Uni-Sign`。
- `Auslan-Daily/`、`MM-WLAuslan/`：数据集。

Drive 布局（`MyDrive/`）：
- `unisign/`：代码。
- `auslan_work/`：
  - `manifest.jsonl`（B/C 联合训练用）、`manifest_mmwlauslan.jsonl`（当前 MM-WLAuslan-only 抽取用）
  - `pose/chunk_*.tar`、`quality.csv`、`excluded.txt`、`overlays_mirrored/`
  - `train_configs/`：自动生成的训练配置
  - `runs/<run>/`：`checkpoints/`、`checkpoint.pt`、`metrics.json`、`predictions.jsonl`、`train.log`

## 6. Arm A 之后的路线

### 6.1 先看噪声底线
两个种子分数的差距就是噪声底线。之后任何改动，都要在每个子集上超过这个差距才算有效。

### 6.2 下一步在两条路里选一条

**(a) 原计划的 Arm B/C**
- **MM-WLAuslan 姿态抽取已完成**：`colab_setup.ipynb` 第 6 节使用 MM-WLAuslan-only 模式和独立的 `manifest_mmwlauslan.jsonl`，设置为 `MMWL_SPLITS = ['Train', 'Valid', 'Test_STU']`。`manifest.py` 的相机过滤明确为 `kf`（Kinect Front），排除了 `Test_MTV` 的 phone/web 多视角数据，也没有复制 Auslan-Daily 视频到 Colab。
- 实际抽取完成 51,440 段（约 4.12M 帧）：压缩包先复制到 Colab 本地盘，由 14 个 worker 并行抽取；每 5 分钟把新增 `.npz` 打成 tar 分片写回 Drive。最终输出 `51,440/51,440 MM-WLAuslan clips extracted`。
- 由于此前 Auslan-Daily 姿态已经在 Drive，重跑 `colab_setup.ipynb` 时仍会扫描已有 tar 分片，但恢复 cell 只解压当前 MM-WLAuslan manifest 对应的 `.npz`；旧的 Auslan-Daily `.npz` 不会被恢复、统计或再次打包。该次准备、数据定位、manifest、archive staging、pose restore、extraction 和 gate cells 均已运行完成；如果之后中断或需要补跑，仍可从前面的准备 cells 重跑，再回到 extraction cell 续上。
- 注意：MM-WLAuslan 的正面相机标识已经根据 release 的目录和文件命名核对为 `kf`，不是凭目录名猜测。
- **之前的 Arm A 没有抽取 MM-WLAuslan**：当时 `MMWL_SPLITS = []`，manifest 只有 Auslan-Daily。后来复制到 Colab 的 MM-WLAuslan 压缩包只是 archive staging，不代表已经生成姿态；本次 MM-WLAuslan 姿态抽取已由第 9 节 worker cell 完成。
- **五个 test 的处理**：当前只抽 `Test_STU`，因为它是与 `Train/Valid` 最接近的 studio 测试集，且当前目标是正面单机位迁移。`Test_ITW`、`Test_TED`、`Test_SYN` 留作后续跨场景泛化评估；`Test_MTV` 是 phone/web 多视角数据，暂不纳入当前 frontal pose-only 流程。它们都不能混入训练。
- **联合 manifest 的关系**：`manifest_mmwlauslan.jsonl` 只服务于当前抽取；B/C 训练仍使用同时包含 `auslandaily` 和 `mmwlauslan` 行的 `manifest.jsonl`。两类 UID 分别以 `ad-` 和 `mmwl-` 开头，可以在需要时拼接，不需要重新抽 Auslan-Daily。

**(b) 脸部通道支线**（用户在另一个讨论里提出的方案，已按代码核对修正）
1. **jaw drop 解耦**：把内唇表示成相对下颌的形变。不改网络结构。
2. **头姿去旋转**：用下颌 9 点 + 鼻尖解相似变换，把脸部点变换到统一参考系。
   - **1 和 2 都会偏离 Uni-Sign 预训练时的脸部输入分布**，所以只能当消融实验，不能默认打开。
   - 要做成配置里的一个开关，并记进训练运行的指纹，不能直接改 `load_part_kp`。
3. **脸部支路扩到 52 点**（外唇 12 + 眉毛 10 + 眼睛 12），给语法性非手动标记和 mouthing 提供输入。
   - 代价比最初估计的小：只有 6 个邻接矩阵要重新初始化，卷积权重可以保留。真正的问题是输入分布变了。
   - 做之前要先把指纹拆成"抽取层"和"选点层"，否则会被当成要重新抽取。**不需要重新抽取。**
   - 只有 1 和 2 显示脸部确实有信号时才做。
4. **眉眼区域 RGB 支路**（零初始化 gate）：推理时会多一套权重。
5. **AV-HuBERT viseme 蒸馏**：依赖 3。AV-HuBERT 只在训练时当离线教师，推理时不加载。

**CTC 前提**：Uni-Sign 没有 CTC 头，也不用 gloss。要加 CTC，得先确认 Auslan-Daily 的 "Sign Language Alignment" 表和 ISLR 表里，gloss 的覆盖率和对齐粒度够不够。Johnston Auslan Corpus 要单独申请访问，写进计划前先确认能不能拿到。

**原则**：规模撑不住堆机制。每加一个机制都要能单独消融，并且和噪声底线比。

### 6.3 其他备选
- 用 `how2sign_pose_only_slt.pth` 或 `openasl_pose_only_slt.pth` 做起点，各跑一次 Arm A 做对比。**OpenASL 已完成（2026-09-17），明显优于 CSL；How2Sign 待跑。** **不要再改 `colab_train.ipynb` 的 `INIT_CKPT`**：已经有 `colab_train_how2sign_init.ipynb` 和 `colab_train_openasl_init.ipynb` 两个专用 notebook，参数对齐官方 Stage3，各有自己的 run 目录，详见第 3 节。
- Arm D：看 OOV 率再决定。

## 7. 待办和已知问题

- **`colab_setup.ipynb` 的检查 cell source 已使用 `--exclude-out`**；如果 notebook 中显示的是旧输出，先重新运行前面的 MM-WLAuslan-only manifest、archive staging 和 restore cells，再运行检查。当前写法：
  ```python
  r = subprocess.run([sys.executable, 'verify_pose.py', '--npz-dir', POSE_LOCAL,
                      '--csv', f'{WORK}/quality.csv', '--exclude-out', f'{WORK}/excluded.txt',
                      '--overlay-dir', f'{WORK}/overlays_mirrored',
                      '--overlay-flag', 'MIRRORED_OR_BACK_VIEW', '--overlay-n', '131'],
                     cwd=CODE, capture_output=True, text=True)
  ```
- **最终测试集评估不能套用排除名单。** test 里有 12 段被排除；报最终成绩要用完整 test 集，才能和文献对比。现在的训练配置只用 train 和 val，暂时不受影响；做 test 评估那一步时要让它默认不套用 `exclude`。
- **News 的姿态标注 `News_Annotation.zip`（6.9 GB）**：在数据集作者的共享 Drive 里，因为下载配额一直打不开。可以右键「制作副本」到自己的 Drive。用途只是验证 News 子集的选人正确率，不急。
  - 文件夹：https://drive.google.com/drive/folders/1Tv2txU3WeW_KHokm9u0s8AU_VDx7cfys
  - 文件 id：`19xBLUbE5GTx1v2BMsSDnBNpOx0r5o3o0`
  - Communication 对应的是 `SP_Annotation.zip`，文件 id `1K3SYjceVJga_3DETUrF-s3bBXz-J7KWq`
- **131 段镜像片段的叠加视频**：用户已检查其中的 `ad-communication-video_29_97.mp4`，确认该片段的姿态跟踪明显错误；当前仍需在 B/C 训练前确认 `excluded.txt` 没有被后续 MM-WLAuslan 检查覆盖。
- 以后在本地改 `spec.py`：只在这一个文件里改，并升级 `SPEC_VERSION`。

## 8. Colab 上踩过的坑（排错参考）

- rtmlib 会顺带装上 CPU 版 onnxruntime，所以要用 `--no-deps` 装 rtmlib。
- onnxruntime-gpu ≥1.27 需要 CUDA 13，≤1.26 对应 CUDA 12。必须装 `[cuda,cudnn]` extras，并把 nvidia 的 lib 目录加进 `LD_LIBRARY_PATH`。`colab_setup.ipynb` 第 2 节已处理。
- Colab 会不打招呼地回收运行时，有一次跑了约 8 小时就被收了。所以训练必须带 `--resume`，笔记本也已经这样写。
- 共享 Drive 文件会被下载配额卡住。gdown 碰到配额页面会误报"下载完成"，只能以字节数加上能否打开 zip 为准。
- 在 VS Code 里编辑一个正开着的笔记本，会导致 `controller is DISPOSED`。
