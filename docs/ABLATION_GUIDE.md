# 结构消融实验指南（Ablation Guide）

> 本文规定论文里**四个核心结构消融**怎么做、每个实验验证什么、哪些东西绝对不能动。
> 执行入口：`scripts/run_ablation_suite.py`（一条命令串行跑完全部）。
>
> 配套：模型/数据集/指标定义见 `docs/EVAL_PROTOCOL.md`；通用评测方法论见 `docs/EVAL_GUIDE.md`。

---

## 0. 要验证的四个设计

以当前完整模型为唯一基准（**Full Model**），分别关掉一个结构设计，看各自有没有实际贡献：

$$
\boxed{
\text{Persistent State}
+\text{Edge-State Conditioning}
+\text{Diffusion Generation}
+\text{Whole-Branch Readout}
}
$$

| # | 实验 | 关掉什么 | 回答的问题 |
|---|---|---|---|
| 1 | **Reset-H** | 跨 reverse timestep 的节点状态记忆 | 跨 diffusion timestep 的图信息记忆有没有用？ |
| 2 | **No-Edge-State** | $z_t \to$ edge state $\to$ GraphFlow 这条反馈 | 当前 branch 假设是否需要反过来调制图信息传播？ |
| 3 | **Direct Prediction** | 整条扩散链 | 为什么需要迭代式扩散，而不是一次 GNN 直接分类？ |
| 4 | **First-Node Readout** | 整段 branch 的 mean pooling | 选 branch 时为什么要看整段道路，而不只看出口第一个节点？ |

---

## 1. 五个 run 的定义

| run | persistent_state | use_edge_state_conditioning | branch_readout | generation_mode |
|---|---|---|---|---|
| **ab_full** | true | true | mean_pool | diffusion |
| **ab_reset_h** | **false** | true | mean_pool | diffusion |
| **ab_no_edge_state** | true | **false** | mean_pool | diffusion |
| **ab_first_node** | true | true | **first_node** | diffusion |
| **ab_direct** | true | true | mean_pool | **direct** |

`ab_full` 是唯一基准。前三个 + `ab_first_node` 是**单因素消融**；
`ab_direct` 是 **non-diffusion baseline**，天然不是单因素（去掉扩散就没有 $z_t$，
也就没有动态 edge state），论文里必须单独说明，不要包装成和前三个一样的纯单因素消融。

### 一个有用的额外对照

`ab_direct` 与 `ab_no_edge_state` 相比，**边状态都是静态的**，唯一差别是
"迭代 50 步"还是"一次前向"。所以这一对实际构成了扩散本身的干净对照。

⚠️ 但两者算力不匹配：Full 的 GraphFlow 调用次数 $= T \times \text{flow\_steps} = 50$，
Direct $= 1$。审稿人一定会问"是不是只是给了 diffusion 50 倍算力"。
**本轮按 as-is baseline 做**，但报告必须给出 GraphFlow 调用次数与单条 query 推理耗时；
补充实验 `Direct-Recurrent`（把静态 GraphFlow 迭代 50 轮）留到第二轮。

---

## 2. 公共规则（不遵守就没法比较）

所有 run 必须完全一致，**唯一变量是被消融的那一个设计点**：

| 项目 | 值 |
|---|---|
| 训练 / val / test 数据 | `data/didi/graph/chengdu/{train,val,test}.pkl`（主训练集） |
| 数据预处理 / Branch Segment | 完全不动 |
| $T$ | 50 |
| `flow_steps` | 1 |
| `d_model` / FFN / dropout | 128 / 256 / 0.0 |
| Edge Cost | 保持 Full 设置（`use_edge_cost: true`） |
| Loss | 完整第四版 loss，**一个字都不改** |
| Optimizer / LR | adamw / 1e-4 |
| batch size | 8（与主实验 M0 实际使用的一致） |
| epochs | 20 |
| `eval_every` | 1（每轮验证，否则会漏掉峰值） |
| best 选择 | `path_similarity_score` / `max` |
| Decoder | strict multi, top_k=2, beam=3 |
| 推理 | deterministic |
| 随机种子 | 0（与之前训练一致；第一轮先看效应量） |

**所有结构消融必须重新训练。** 不许拿 Full 的 checkpoint 在推理时临时关模块充当消融结果。

### 为什么是 20 epoch

主实验 M0 的 val 曲线（`eval_every=5`）：

| epoch | 5 | 10 | 15 | 20 | 25 | 30 | 35 |
|---|---|---|---|---|---|---|---|
| PathSim | 0.3971 | 0.4024 | **0.4145** | 0.3878 | 0.3943 | 0.3961 | 0.3816 |

第 5 轮就到 0.397，之后 30 轮都在 0.38–0.41 之间震荡 —— 不是过拟合，是**收敛后在小范围抖**。
所以 100 epoch 纯属浪费，20 轮足够覆盖峰值。若某个变体（尤其 Direct）收敛更慢，
`best.pt` 会自己选到合适的那轮；真不收敛再续训即可（`--extra-epochs`）。

---

## 3. 实现细节（改动落在哪里）

四个开关都加在 `model.*`，**默认值等于改造前的行为**：

```yaml
model:
  persistent_state: true
  use_edge_state_conditioning: true
  branch_readout: mean_pool     # mean_pool | first_node
  generation_mode: diffusion    # diffusion | direct
```

| 开关 | 落点 | 要点 |
|---|---|---|
| `persistent_state` | `src/training/losses.py`（训练整链）、`src/diffusion/sampler.py`（推理链） | **训练和推理必须同时生效**。开关在**循环里**，`model.step` 本身不知道 —— 所以单测必须跑完整链、检查每次 step 收到的 $H_t$ |
| `use_edge_state_conditioning` | `src/models/denoiser.py::_edge_features` | 改用 `static_edge_state_ids()`：只有 source-forced 边 selected，**与 $z_t$ 完全无关**。EdgeStateEncoder 的参数保留，所以输入维度与参数量都不变 |
| `branch_readout` | `src/models/branch_scorer.py::branch_representation` | 候选集合、$z_t$ 语义、edge state 展开、decoder **全部不动**，只改"这个 branch 怎么表示"。输入仍是 $3d$，参数量不变 |
| `generation_mode` | `src/models/denoiser.py::direct_logits` + `src/training/losses.py::direct_prediction_loss` + `src/diffusion/sampler.py::direct_chain` | `direct_logits` **签名里没有 $z$**，从结构上杜绝 label leakage；时间条件固定 $t=0$，参数量不变 |

### 最容易做错的三件事

1. **只在推理阶段 reset-H** —— 不行。训练也必须是 reset-H，否则训练看到的是 persistent、
   测试突然 reset，结果没有意义。
2. **把 reset 做成 `H_t.detach()`** —— 这是另一个实验。`detach` 保留上一轮**数值**、只切梯度；
   reset 是**连数值都不要**。代码里这两行恰好相邻（`losses.py` 里紧跟 `truncate_every`），极易改错。
3. **消融顺手改了别的东西** —— 例如做 First-Node 时把候选也改成"第一个节点"、
   或把 edge state 只标到第一跳。那就不再是单因素消融了。

---

## 4. 执行

### 一条命令跑完全部（推荐）

```bash
python scripts/run_ablation_suite.py
```

串行执行：**训练 5 个 run → 每个 run 评测 3 个测试集 → 出报告**。
已有产物自动跳过，可随时中断再跑。

### 分阶段

```bash
python scripts/run_ablation_suite.py --stage train    # 1) 只训练（约 22 小时）
python scripts/run_ablation_suite.py --stage eval     # 2) 只评测（约 1 小时）
python scripts/run_ablation_suite.py --stage report   # 3) 只出报告
python scripts/run_ablation_suite.py --dry-run        # 打印将执行什么，不真跑
python scripts/run_ablation_suite.py --only ab_full ab_direct
python scripts/run_ablation_suite.py --force          # 已存在的也重跑
```

### 产物

```
outputs/runs/ab_full/            best.pt  last.pt  history.json  run_config.json
outputs/runs/ab_reset_h/         ...
outputs/runs/ab_no_edge_state/   ...
outputs/runs/ab_first_node/      ...
outputs/runs/ab_direct/          ...
outputs/ablation/<run>__<dataset>.json    15 份评测结果（5 run x 3 数据集）
outputs/ablation/ABLATION_REPORT.md       自动生成的对比表 + Δ 表
```

每个 run 的 `run_config.json` 都记录了生效的开关，可以自证是哪个变体。

---

## 5. 评测矩阵

每个 run 都测全部三个数据集（**不能只测 normal**）：

| 数据集 | 作用 |
|---|---|
| 成都 normal（`chengdu/test_1000`） | 常规路径能力 |
| **成都 long（`chengdu_long/test_1000`）** | **最重要的机制诊断集** —— 长决策链才能暴露 persistent / edge-state / branch 表示的差异 |
| 西安（`xian/test_1000`） | 结构泛化是否受影响 |

口径与主实验完全一致：**strict multi, top_k=2, beam=3, deterministic**。

> 如果结果是 `normal: Full ≈ Reset` 但 `long: Full >> Reset`，**这是好结果而不是实验失败** ——
> 它恰好说明这个设计是在长决策链上起作用。所以 long 这一行是重点。

---

## 6. 报告指标

每个 run × 每个数据集统一输出：

| 指标 | 方向 | 含义 |
|---|---|---|
| GoalHit | ↑ | 完整走到终点的比例 |
| Broken | ↓ | 路径含图中不存在边的比例 |
| PathSim | ↑ | 归一化 LCS，**没到终点记 0** |
| nLCS | ↑ | 只在成功样本上的归一化 LCS |
| EdgeF1 | ↑ | 无向边集合的 F1 |
| pred/GT | ↓ | 预测路径长度 / 真实路径长度 |
| DTW(km) | ↓ | km 级动态时间规整距离 |

另外**必须报告**（尤其 Direct）：

- 参数量（四个消融**全部不变**，这点要写进论文 —— 直接反驳"只是参数更多"）
- 单条 query 推理耗时
- GraphFlow 调用次数（Full = 50，Direct = 1）

---

## 7. 验收与回归

### 7.1 回归要求（改代码后、训练任何消融之前必须先过）

新增开关**默认状态下必须与改造前的 Full model 行为一致**。

判定方式（已执行，结论：通过）：

```
指标总数 31，其中 28 项逐位相同（含 GoalHit / PathSim / nLCS / EdgeF1 / pred-GT / DTW 全部）
不同的 3 项：mean_elapsed、wall_time、soft_goal_reachability
    -> 用新代码连跑两次，差异项**完全相同**
    -> 忽略 elapsed 后，1000 条逐 query 结果与改动前**逐位相同**
```

即：那 3 项是 GPU/AMP 的运行间抖动，与改动无关。

复现这条回归：

```bash
python scripts/evaluate.py --config configs/didi_chengdu.yaml \
  --checkpoint outputs/runs/didi_chengdu_loss_improved/best.pt \
  --data data/didi/graph/chengdu/test_1000.pkl \
  --out outputs/eval/_regression_check.json --beam-width 3 --device cuda --no-progress
# 应得 GoalHit=0.9960  PathSim=0.4047  Broken=0.0040
rm outputs/eval/_regression_check.json
```

### 7.2 单元测试

```bash
python -m pytest tests/test_ablation_switches.py -q     # 20 个
python -m pytest tests -q                               # 全套 562 个
```

覆盖的关键点：

- 默认开关 == Full；默认 readout 与 `branch_mean_pool` 逐位相同
- `persistent_state=false` 时**每一步**的输入都等于 `init_nodes()`（不是 detach）
- 关掉 edge-state conditioning 后，只换 $z_t$ 输出逐位不变；开着则必须变
- `static_features` 的签名里没有 $z$（杜绝 label leakage）
- First-Node：只改 branch 内部节点的 $H$，first_node 分数不变、mean_pool 必须变
- 四个变体参数量与 Full **完全一致**
- `direct_logits` 签名里没有 $z$；direct 链**一次都不调用** `model.step`
- `direct_prediction_loss` 不碰 diffusion 的任何采样接口

---

## 8. 结果怎么读

四个实验分别对应论文的四句话：

| 实验 | 若 Full 明显更好，支持的说法 |
|---|---|
| Reset-H | *Persistent graph information flow* —— 节点隐状态跨 timestep 持续传播，对长路径尤其重要 |
| No-Edge-State | *Branch state guides information flow* —— 当前 branch 假设反过来调制图上的消息传递 |
| Direct | *The structured decision field is progressively refined through diffusion* |
| First-Node | *Whole-branch representation provides mesoscopic routing context* |

**若某个消融没有明显退化**，那也是一个有价值的结论（说明该设计冗余），
照实写，不要为了凑故事去改数据或口径。
