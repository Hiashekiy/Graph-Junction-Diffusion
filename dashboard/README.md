# Interactive dashboard

启动面板（建议使用项目 README 中配置的、已安装 PyTorch 的解释器）：

```bash
E:/CondaEnvData/envs/GGMPC/python.exe dashboard/server.py
```

默认打开 `http://127.0.0.1:8765/`。如不希望自动打开浏览器：

```bash
E:/CondaEnvData/envs/GGMPC/python.exe dashboard/server.py --no-browser
```

面板会自动发现：

- `outputs/runs/**/run_config.json + best.pt` 中的模型；
- `data/**/*.pkl` 中的数据集（按来源分组：`controlled` / `long` / `oldv1` / `mixed` / `smoke`）；
- run 目录中的 `eval*.json` 和 `mp_*.json` 指标。

数据集下拉按上面那套目录分组显示（`optgroup` 标签 = 子目录名 + 条目数），鼠标悬停在选项上
能看到仓库内的相对路径（如 `data/controlled/controlled_test.pkl`）。旁边的「随机」按钮会在
**当前数据集**里随机抽一条样本并直接生成，用来快速浏览数据；它会避开刚看过的那条，切换
数据集时可用范围也会跟着刷新。

> 数据集目录/文件改动后记得**重启 server**：清单是启动时扫的（`discover_datasets`），
> 跑着的旧进程会一直用启动时那份，路径失效后点生成只会得到 "推理未完成"。

路径页会调用项目真实的扩散采样及单路径/多分支解码。首次选择一个模型时需要加载
checkpoint，之后同一模型会复用内存缓存。切换模型时旧模型会释放，避免多个模型同时占用
GPU 显存。

多分支路线使用后端提供并校验的物理边序列。共享边上的彩色线路只在边的内部做平行偏移，
在真实节点中心汇合，所以相邻两段永远共用同一个端点，不会出现悬空短线或错位接头。

**公共前缀不铺车道**：一条边如果被当前显示的**全部**路线经过（`uniqueRoutes == totalRoutes`），
它不携带任何区分信息 —— 现在合并成一条中性色主干线（`TRUNK_COLOR = #93a3bd`），
只有**真正分叉之后**的边才按路线各自偏移着色。在这之前 8 条路线会把公共前缀画成 8 条
平行细线，看上去像"一开始就分叉了"（实测反馈）；左侧提示语与图例也说明了这一点。
线路与非路径节点发生视觉交叉时使用底色衬线跨过，只有真正属于路线的节点才覆盖在线路
上方，避免把普通交叉误认为连接或缺线。

播放进度**不**使用 `stroke-dasharray`，而是由 `app.js` 在渲染后对每条边做等弧长采样
（`getTotalLength` / `getPointAtLength`，元素必须已经挂到文档里），再按当前进度用采样点
重写 `d`，游标头直接取截断点。原因是一个真实踩过的坑：只要描边带
`vector-effect: non-scaling-stroke`，Chromium 会把 `stroke-dasharray` 当成**屏幕像素**
而不是用户单位来解释，而 `getTotalLength()` 仍返回用户单位；于是「长度等于总长的 dash」
只能画出整条边的 `1 / scale`（面板常见缩放 1.19 时约 84%），每条边在进入下一个节点前
都会缺一截，看起来就是路径快到节点时突然断开。现在 `styles.css` 里 `.route` /
`.route-casing` 不再声明 `vector-effect`，描边粗细与节点半径一样使用用户单位，随面板
一起缩放。采样结果逐帧增量追加，回放/重播时才重建前缀。


## 加权模型与多分支解码（新）

面板现在把 Weighted 扩展与多分支增强一并展示出来：

- **模型徽章**：按 run 的 `data.weighted` 与 `model.use_edge_cost` 标 **带权** / **带权·无cost** / **无权**；
  目前带权的是 `v2_weighted_controlled`，消融对照是 `v2_weighted_controlled_cost_ablated`。
- **口径**：一次评测展开成 `single` / `multi` / `multi · best_goal` / `multi · best_goal_cost` /
  `multi k=N / stop|skip`，与方案的「三条口径」一一对应；新增两列 **W-opt cov**（加权最优覆盖率）
  与 **Goal paths**（平均 Goal 路径数）。
- **数据集**：`data/weighted_controlled/*.pkl`（w ~ U(1,10)）与无权数据集并列显示；评测产物的数据集
  归属优先取 JSON 里记录的 `data` 字段（老产物按 `data.weighted` 兜底），不再靠文件名猜。
- **路径页**：多分支控制条多了「必死 branch 预筛选」（透传 `filter_dead_branches`），summary 显示剔除数量；
  带权图上每条路线同时显示跳数与**真实 cost**。

## 实验报告页签（新）

`GET /api/reports` 给出固定清单，`GET /api/reports/<id>` 返回内容：Markdown 直接渲染
（标题/表格/列表/代码块），JSON 汇总格式化展示。清单：

| id | 文件 |
|---|---|
| `report` | `docs/REPORT_multipath_and_weighted.md`（全模型统一评测报告） |
| `all_models` | `outputs/all_models_multipath_summary.json` |
| `weighted_experiment` | `outputs/weighted_experiment.json`（加权 vs cost 消融） |
| `multipath_weighted` / `multipath_filteron` | 过滤 on/off 的 beam=64 对照 |
| `beam_compare` | beam=64 vs beam=3（CPU） |
| `regression` | 零破坏回归：旧 checkpoint 逐位复现 |

> 报告清单与 run/dataset 清单都在**启动时**扫描，新增产物后重启 server 才会出现在列表里。

### 三个状态不是一回事（#92 那个疑问）

路径页上同时能看到三个东西，**它们不是同一个状态**，面板把差异显式写出来：

| 状态 | 是什么 | 谁在用它 |
|---|---|---|
| **final argmax readout** | 最后一个 reverse step 的 `candidate_prob` 做**组内 argmax** | **single 解码用的就是它**（第 23 节的口径变更） |
| **链状态 `z_{t-1}`**（扩散视图 noisy 档） | 扩散链**自己携带**的状态（默认模式=按 posterior 采样） | **只作诊断**，single 解码不再跟随它 |
| **模型预测 `ẑ₀`**（扩散视图 clean 档） | 模型每一步对干净状态做的 argmax 预测（每步的，不是最终 readout） | 只作诊断 |
| **全程 argmax rollout**（`stochastic=False`） | 每一步 posterior 都取 argmax 走到底 | 面板「全程 argmax（对照）」开关 |

实测 #92（`weighted_controlled_test.pkl`，`v2_weighted_controlled`，seed 0，CPU）：

```
GT                                  18 跳
readout z0 = [2,5,10,16,20] -> goal    18 跳 / cost 101.29   （single 解码用这条）
sampled z0 = [2,5,10,16,19] -> broken  16 跳 / cost  92.35   NULL selected at 8（仅诊断）
唯一分歧在 decision 4（node 8）：readout 选 ->9，采样抽到 NULL
```

也就是说：**single 解码不再读取采样状态**，所以 #92 的默认 single 结果现在是"到达"；
扩散视图的"链状态"档仍然显示那次采样（中止），这是正常的 —— 两个口径本来就不同。
面板为此提供：

- 左侧对照框：「Single 解码 = 最终 candidate_prob 的组内 argmax」+ 解码所用的状态与链状态并排；
- 控制条上的 **全程 argmax（对照）** 开关：切到"每一步都取 argmax"的 rollout（与默认 single 不同）；
- 扩散帧标题与明细行标出 `clean→… / noisy→…`。

生成完成后可以在图上切换到“扩散过程”。时间轴记录完整的 `T -> 0` reverse chain，并可
在两种真实中间状态之间切换：

- `预测干净状态 z_hat_0`：每一步 Branch Scorer 对 clean state 的组内 argmax；
- `去噪后带噪状态 z_(t-1)`：该步 reverse posterior 完成采样后的实际状态。

每一帧显示完整 decision field：每个分叉节点严格选择一个 Branch 或 NULL。

- **选到 Branch 的分叉点会亮起**：节点描边与填充换成当前模式颜色并加光晕，外圈标记本轮
  选择的发起点；**选 NULL 的分叉点会调暗**（`opacity: .22`），因此一眼就能看出这一步到底
  哪些路口真的做了动作。NULL 的数量同时写在帧信息里，不额外绘制虚线圈。
- **每条 Branch 的每一条物理边都带一个三角箭头**表示方向。箭头不固定在边的某个比例上，
  而是在边的 30%–70% 区间里挑一个离所有节点最远的落点：节点圆是画在线路之上的，固定比例
  经常被节点盖住——这也是之前"有些 branch 看不到箭头"的原因。
- 选择发起点使用外圈标记，因此其他 decision 的 Branch 汇入同一节点时，不会被误认为该节点
  同时选择了两条出路。

拖动时间轴可检查任意一步，也可以自动播放全部扩散过程。
