# Interactive dashboard

启动面板（建议使用项目 README 中配置的、已安装 PyTorch 的解释器）：

```bash
E:/CondaEnvData/envs/GGMPC/python.exe dashboard/server.py
```

默认打开 `http://127.0.0.1:8765/`。如不希望自动打开浏览器：

```bash
E:/CondaEnvData/envs/GGMPC/python.exe dashboard/server.py --no-browser
```

面板有两个页签：**路径可视化** 与 **实验报告**。

> 2026-09-16：原来的「测试指标可视化」页签（指标柱状图 + 指标明细表 + 模型筛选 +
> 汇总栏）连同后端 `discover_metrics()` 一起删掉了。历史 `eval*.json` 里那批口径
> （`coverage_rate` 的 top-k 逃逸、`optimal_coverage_rate` 的 beam 预算效应、KLEV 的
> `log(1/eps)` 依赖）在 README 里都有记录，继续在面板上并列展示只会让人拿它们做
> 跨配置比较。要看数值请直接读 `outputs/reports/*.json`。

## 面板会自动发现什么

- **模型**：`outputs/runs/**/run_config.json + best.pt`。当前有三个：

  | run | 类别 | 数据 |
  |---|---|---|
  | `controlled_unweighted` | 无权 | 合成 controlled 图 |
  | `controlled_weighted` | 带权 | 合成 controlled 图 + 道路长度 |
  | `didi_chengdu` | **滴滴·带权** | 成都真实路网 + 真实车辆历史路径当 GT |

- **数据集**：`data/**/*.pkl`，按**相对 `data/` 的目录**分组：
  `unweighted` / `weighted` / `didi/graph/chengdu`。下拉项的 title 给出仓库内相对路径
  与体积。

  三类 pkl **刻意不列**（用 `GraphQueryDataset.load` 打开会直接抛异常，而面板只会在
  点下"生成"之后才报一句"推理未完成"）：

  * `data/didi/raw/**` —— `dicts.pkl` / `ChengDu.pkl` 是模型**输入**，不是数据集；
  * `_` 前缀 —— `_didi_candidates.pkl` 是 `prepare_didi.py` 的中间缓存；
  * `graph_global.pkl` —— 整城 junction 图，corridor 就是从它切出来的。

> 数据集目录/文件改动后记得**重启 server**：清单是启动时扫的（`discover_datasets`），
> 跑着的旧进程会一直用启动时那份，路径失效后点生成只会得到 "推理未完成"。

## 滴滴真实数据 + 街道底图

选中 `didi_chengdu` 时，路径页会切成**真实地理模式**：

- 浅灰细线 = 整张成都路网（OSMnx，2891 节点 / 4403 边）；稍亮一档的蓝灰线 = 该样本的
  OD corridor 子图（模型真正看到的那张图）；彩线 = 模型解出来的路线。
- **形状不会被拉伸**：经度方向按 `cos(lat0)`（成都 ≈ 0.81）校正后与纬度同尺度，
  再按面板长宽比 letterbox。后端 `PANEL_WIDTH/HEIGHT/MARGIN` 与前端
  `GRAPH_WIDTH/HEIGHT/MARGIN` 必须一致，`tests/test_dashboard.py` 里有一条静态比对
  钉住这件事 —— 两边一漂移，城市就会被横向压扁。
- **裁剪**：先按面板比例把 corridor 包住，再整体放大 15% 露出周边街道。顺序不能反 ——
  先加 margin 再补比例，会在 1.75:1 的宽屏上把近乎方形的 corridor 压成中间一小块
  （实测只剩 40% 宽）。
- **比例尺**：左下角一条 `N km / N m` 的标尺，由后端算出的 `km_per_x_unit` 换过来。
  没有它，"真实经纬度底图"就只是一堆灰线，看不出这段路是 300 米还是 3 公里。
- 图下方一行说明底图来源、坐标覆盖率、画面多少公里、街道条数。
- 节点不再全画：地理模式下一个 corridor 有 100~350 个节点、绝大多数是 decision，
  全画成 r=8 的圆会糊成一团。现在普通节点不画、decision 画小圈、只标 S/G。

合成数据集没有地理位置，自动退回弹簧布局（`graph.geo = false`），行为与改动前一致。

### 数据集下拉会自动跟着模型走

模型下拉与数据集下拉是独立的，但选错组合（例如 `didi_chengdu` + `unweighted_test.pkl`）
必然失败：图和特征维度都对不上。所以切换模型时，如果当前数据集**不在**该模型的
`data_dir` 下，面板会自动切到那一份最合适的 split（优先 `test_1000`，再退 `test`），
并用一条中性提示条说明。反过来手动改数据集不会被覆盖。

## 过期的 run 快照（踩过的坑）

`outputs/runs/<run>/run_config.json` 是**训练当时**的快照，仓库重构后可能指向早就删掉的
路径。实测 `didi_chengdu` 那份里写着：

```
data.root        = data/DiDiChengduXian/didi_datasets/datasets/didi_chengdu
data.coords_file = data/DiDiChengduXian/data/data/cd/ChengDu.pkl
paths.data_dir   = data/didi_chengdu_gjd
paths.run_name   = didi_chengdu_flow1_weighted_new
```

四个名字/路径在 2026-09-16 的清理里全失效了（真实位置是
`data/didi/raw/chengdu/ChengDu.pkl` 与 `data/didi/graph/chengdu`）。所以职责拆开：

- **模型结构 / 扩散参数** → 必须用 `run_config.json`（权重兼容，`_bundle()` 用的就是它）；
- **数据在哪、坐标在哪** → 一律用**当前** `configs/<run>.yaml`（`_live_run_settings()`）。

名字的查找顺序是"**目录名优先**，再退到快照里的 `run_name`"，最后还会扫一遍
`configs/*.yaml` 里 `paths.run_name` 的声明。合并/改名过的 run 目录留着的是旧快照，
只有目录名才是当前配置名。

## 路径页会调用真实推理

路径页调用项目真实的扩散采样及单路径/多分支解码。首次选择一个模型时需要加载
checkpoint，之后同一模型会复用内存缓存。切换模型时旧模型会释放，避免多个模型同时占用
GPU 显存。数据集也最多缓存两份，防止把几百 MB 的 pkl 全留在内存里。

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

> 街道底图 `.street-edge` **例外**：它没有 `stroke-dasharray`，所以用
> `vector-effect: non-scaling-stroke` 是安全的，而且正是想要的 —— 底图在放大时应该保持
> 发丝粗细，而不是跟着涨成一片灰。

## 实验报告页签

`GET /api/reports` 给出固定清单，`GET /api/reports/<id>` 返回内容：Markdown 直接渲染
（标题/表格/列表/代码块），JSON 汇总格式化展示。清单只列**当前仓库里真的存在**的产物
（`docs/ARCHITECTURE_V2.md`、`docs/Graph-Junction-Diffusion_V2_代码实施指南.md`、
`outputs/reports/*.json`、以及零破坏回归产物）。

被删掉的 `docs/REPORT_multipath_and_weighted.md` 已从清单移除 —— 留一个恒
`exists: false` 的条目只会让报告页多一个死链接。

> 报告清单与 run/dataset 清单都在**启动时**扫描，新增产物后重启 server 才会出现在列表里。

## 三个状态不是一回事（#92 那个疑问）

路径页上同时能看到三个东西，**它们不是同一个状态**，面板把差异显式写出来：

| 状态 | 是什么 | 谁在用它 |
|---|---|---|
| **final argmax readout** | 最后一个 reverse step 的 `candidate_prob` 做**组内 argmax** | **single 解码用的就是它**（第 23 节的口径变更） |
| **链状态 `z_{t-1}`**（扩散视图 noisy 档） | 扩散链**自己携带**的状态（默认模式=按 posterior 采样） | **只作诊断**，single 解码不再跟随它 |
| **模型预测 `ẑ₀`**（扩散视图 clean 档） | 模型每一步对干净状态做的 argmax 预测（每步的，不是最终 readout） | 只作诊断 |
| **全程 argmax rollout**（`stochastic=False`） | 每一步 posterior 都取 argmax 走到底 | 面板「全程 argmax（对照）」开关 |

实测 #92（`weighted_test.pkl`，`controlled_weighted`，seed 0，CPU）：

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

生成完成后可以在图上切换到"扩散过程"。时间轴记录完整的 `T -> 0` reverse chain，并可
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
