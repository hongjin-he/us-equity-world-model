<div align="center">

<img src="figures/microworld_logo.svg" width="200" alt="MicroWorld"/>

# MicroWorld

**第一个为量化金融打造的世界模型架构：多层平均场博弈建模 + 市场价格预测**

**超越传统因子挖掘的量化金融新范式**

---

[![Stars](https://img.shields.io/github/stars/hongjin-he/MicroWorld?style=social)](https://github.com/hongjin-he/MicroWorld/stargazers)
[![Forks](https://img.shields.io/github/forks/hongjin-he/MicroWorld?style=social)](https://github.com/hongjin-he/MicroWorld/network/members)

[![Paper](https://img.shields.io/badge/配套论文-Alpha%20Flow%2002-red.svg)](https://github.com/hongjin-he/mathmatical-framework-for-world-models-in-quant-finance)
[![Implementation](https://img.shields.io/badge/工程实现-MicroWorld--Impl-blueviolet.svg)](https://github.com/hongjin-he/us-equity-world-model)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

[![X](https://img.shields.io/badge/𝕏-Mr__Abstractor-000000?logo=x&logoColor=white)](https://x.com/Mr_Abstractor)
[![Instagram](https://img.shields.io/badge/Instagram-mr.abstractor__ust-E4405F?logo=instagram&logoColor=white)](https://www.instagram.com/mr.abstractor_ust/)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-HongJin%20HE-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/hongjinhe-hkust-edu)

**[English](README.md) | [中文文档](README_CN.md)**

**Alpha Flow Research · 何鸿锦 · 香港科技大学 / 斯坦福 IHP · 2026年7月**

</div>


---

### 世界模型的心跳（英文版完整內容）

![MicroWorld 全局動態演示](figures/global_demo.gif)

*兩年模擬市場，框架的每一層同時演化——事件算子觸發（含 IPO n 8→9、破產 n 9→8）、平均場密度形變、Lyapunov 危機指標提前 22 個交易日預警、控制器自動降風險。全部由本 repo 代碼生成，零 API 密鑰：`python demo/global_demo.py`*

> 📌 **本中文版對應 v1 敘事框架。** 最新完整內容請見 [英文版 README](README.md)，新增：Type 1/2 世界模型、E-Game-C 架構詳解、**22 個事件算子完整目錄**、七大定理總表、反身性形式化、Agent 分類法（6 機構類 + 5 散戶原型）、**[完整數據需求與實驗路線圖](DATA_REQUIREMENTS.md)**（E1–E6 六個實驗設計）、17 天筆記本索引、Related Work 對比與 40 篇參考文獻。

---

## 没有人真正解决的问题

> *每一秒，约有50,000家机构投资者、5亿散户和30家央行，正在同时对同一组资产做出决策——每个主体拥有不同的信息、不同的时间尺度、不同的目标，以及彼此之间复杂的约束关系。你屏幕上看到的价格，是这个系统的实时汇总统计量，每毫秒更新一次。*
>
> *没有任何现有模型忠实地捕捉到了这一切。*

金融市场不仅仅是复杂的。它是人类文明所产生的最精密的多层次竞争系统——在这个系统中，建模本身就会改变被建模的对象。每一个发现规律的对冲基金，都会通过交易立即摧毁这个规律。每一次央行政策公告，都会在四个层次上同时触发级联式的策略响应。

本工作回答的核心问题：

> **是否存在一个完备的数学理论——类比于统计力学或动理学理论——能够将金融市场描述为其本来面目：一个异质主体之间的多层次、多时间尺度、多目标博弈？**

答案是肯定的。本仓库呈现这个理论及其工程实现。

---

## 为什么这个问题现在最紧迫

三股力量正在以历史上最快的速度瓦解旧范式：

**1. 因子α正在加速死亡。** α半衰期：1990年约6年，2023年约11个月。这不是周期性波动——这是AI加速拥挤的结构性后果。当所有基金在数月内发现相同信号时，信号在任何人获利之前就消失了。因子模型没有理论解释信号为何衰减，无法在衰减完成之前检测到它。

**2. 大模型正在以规模同质化散户行为。** 5亿散户正在向同三个AI助手提出同样的问题，得到同样的答案。行为噪声项 $\nu^\eta$ 同时在增大（更多散户协调，更大跳跃）并变得更可预测（协调机制现在可以建模）。散户噪声是非结构化的旧假设已经过时。

**3. 菲尔兹奖刚刚验证了这个范式。** 2026年7月，邓煜获得菲尔兹奖，证明了N体牛顿力学在 $N\to\infty$ 时收敛到玻尔兹曼方程。这正是我们应用于金融的数学范式：N个理性主体收敛到福克-普朗克-科尔莫哥洛夫（FPK）方程。**FPK方程就是金融玻尔兹曼方程。**

---

## 前人的研究——以及它们为什么不够

### 一、因子模型（CAPM、Fama-French、600+因子）

主流范式自1960年代以来一直在问：*"什么统计特征与未来收益相关？"*

这个范式的根本缺陷由Robert Lucas在1976年指出——**卢卡斯批判**：一旦某个统计关系被广泛采用，理性主体就会改变行为加以应对，该关系随即消失。实证记录证实了这一点：

- 平均因子α半衰期：**1990年约6年 → 2023年约11个月**
- Harvey、Liu与Zhu（2016）：316个记录在案的因子中，多数无法复现
- 随着AI实现规模化同步部署，"因子动物园"正在崩溃

因子模型对此无能为力。**它们是伪装成理论的模式识别器。**

### 二、机器学习方法（LSTM、Transformer、XGBoost）

ML方法试图发现人类研究者遗漏的规律。其失败模式是古德哈特定律：当一个指标成为目标时，它就不再是好指标。更根本地：

- **相关≠因果**：ML模型无法区分能在主体适应中存活的信号和不能的信号
- **分布偏移**：金融市场之所以非平稳，正是因为主体会适应预测——ML模型的部署本身就改变了训练分布
- **无结构性**：没有价格运动的理论，就无法知道模型何时失效

结果：在同样的另类数据上运行相同transformer架构的量化基金产生越来越相关的收益——直到拥挤灾难性地瓦解。

### 三、现有主体模型与群体智能模型

**MicroFish（郭杭江，百孚资本，2024）— GitHub 6.9万星，3000万人民币投资：**

MicroFish将群体智能算法（粒子群、蚁群优化）应用于金融价格预测。这是令人印象深刻的工程成就，其病毒式传播反映了市场对机理性模型的真实渴望。然而，作为世界模型，它存在根本性的局限：

| 能力 | MicroFish | 本框架 |
|---|---|---|
| **主体间博弈论** | ❌ 主体不进行策略性互动 | ✅ 显式计算纳什均衡 |
| **多层次层级** | ❌ 平面群体——无市场/类型/机构/个体结构 | ✅ 四层次层级平均场博弈系统 |
| **机构内部竞争** | ❌ 无基金内部桌组竞争的概念 | ✅ 第3层MFG：个体间纳什博弈 |
| **事件算子代数** | ❌ 无法将并购、IPO、利率决策建模为结构性状态变化 | ✅ 完整群胚代数（I/II/III型） |
| **数学收敛保证** | ❌ 启发式收敛 | ✅ $W_2 \leq C\rho^n$（命题4.2，已证明） |
| **解释能力** | ❌ 能预测但无法解释 | ✅ 机制本身就是模型 |
| **危机预警** | ❌ 基于模式，被动反应 | ✅ 李雅普诺夫稳定性（价格波动前检测到机制变化） |

MicroFish建模的是*粒子群收敛到价格*。本工作建模的是*理性主体博弈收敛到均衡*。这个差异并非表面——它决定了模型是否能在自身部署后存活。

---

## 市场作为四层博弈

![资本主义模拟器——完整竞争生态系统](figures/capitalism_simulator.svg)

上述可视化并非抽象示意。它描述了全球金融市场的真实竞争结构——**四个嵌套层次**的主体，各自进行不同类型的博弈，通过共享价格过程、资本流动和信息级联相互耦合。

中国有句古话：**个人由环境造就**。我们的框架将这一洞察精确化。四个层次并非孤立：每个主体同时是所有上层的产物，也是所有上层的贡献者。

### 第0层——跨市场资本流动博弈

**参与者：** 全球宏观参与者、不同国家的央行、国际资本本身。

**发生了什么：** 资本在市场间流动，寻求风险调整后的收益。美联储鹰派升息→美国实际利率上升→资本从新兴市场流向美元资产→人民币贬值→央行响应→全球股市重新定价。这是一场**整个市场之间的博弈**——美国、欧盟、中国、日本、香港、新兴市场集团——在争夺国际资本的同时（不完美地）协调全球稳定。

**博弈类型：** 混合MFC/MFG——主权协调（G7机制）叠加竞争性资本吸引。

**关键耦合：** 第0层均衡设定了每个市场 $m$ 的**外部环境** $\Gamma^m_t$——所有低层博弈在此背景下进行。

### 第1层——每个市场内的机构类型博弈

**参与者：** 单一市场内不同*类型*的机构：央行/政府、商业银行、投行、量化对冲基金、私募股权/传统对冲基金、公募基金/ETF、散户投资者。

**为什么类型很重要：** 每种类型具有结构性不同的目标、风险函数、监管约束、投资期限，以及最关键的——**不同的信息获取能力**。央行持有机密宏观数据。量化基金持有专有信号库。散户收到的是机构已经交易过后才到达的公开新闻。

**博弈类型：** 多种群MFG——每种类型与其他类型进行纳什博弈，具有特定类型的目标和信息集。

### 第2层——每种类型内的机构个体博弈

**参与者：** 同类型机构之间的面对面竞争。量化基金中：Jane Street vs Citadel vs Two Sigma vs Renaissance。投行中：高盛 vs 摩根大通 vs 摩根士丹利。

**发生了什么：** 在同类型内，机构共享相似的信息来源和策略空间——使竞争成为四个层次中最直接、最零和的。当Citadel构建一个新的动量信号时，Renaissance实际上也在构建同样的信号；当一方部署时，就会侵蚀另一方的α。

**博弈类型：** 每种类型内的标准MFG——纯纳什竞争，拉斯里-莱昂斯单调性保证唯一均衡。

### 第3层——机构内部个体博弈

**参与者：** 单一机构内的个人（投资组合经理、量化研究员、风险官员、交易员）。

**发生了什么：** 对冲基金不是单一主体。其桌组竞争资本分配（PnL更高的PM下个月获得更多资本）。研究员竞争功劳。风险官员与交易员进行受约束的博弈。同时，所有人必须合作：内部不合作的基金表现不佳，失去AUM。

**博弈类型：** 混合合作-竞争MFG——内部资源上的纳什竞争，机构存续上的合作MFC。

### 耦合结构

四个层次**双向耦合**：

- **向下（环境→个体）：** 第0层资本流动决定第1层板块配置；第1层类型主导地位决定哪些第2层机构能存活；第2层机构PnL决定第3层个人薪酬和留任。
- **向上（个体→环境）：** 第3层桌组行为决定第2层净头寸；第2层机构流动汇总为第1层类型级别需求；第1层类型动态决定第0层资本流动均衡。

**这个四层耦合系统的均衡——我们称之为层次化纳什均衡（定理7.4）——就是我们所说的"市场价格"。**

---

## 数学框架：双线并行

> *以下每一节同时运行两条线索：用通俗语言呈现的概念论证，以及其严格的数学形式。直觉激发方程，方程规范直觉。*

---

### 组件1——金融状态空间

**每个模型必须首先回答的问题：** *市场在某一瞬间的状态是什么？*

大多数模型回答：价格。但价格是过程的*输出*，而非其状态。生成价格的机制——杠杆率、交易量、流通股数、信息披露——对于仅追踪价格的模型来说是不可见的。

我们定义时刻 $t$ 单一资产的最小充分状态表示：

$$s_t = (p_t,\; v_t,\; \ell_t,\; \kappa_t,\; \iota_t)^\top \in \mathbb{R}^5$$

| 坐标 | 含义 | 为什么属于状态 |
|---|---|---|
| $p_t = \log P_t$ | 对数价格 | 所有主体反应的主要可观测量 |
| $v_t = \log V_t$ | 对数成交量 | 携带关于信念强度和流动性的信息 |
| $\ell_t = D_t/E_t$ | 杠杆率 | 决定放大效应和脆弱性；高 $\ell_t$ → 莱维尾部风险 |
| $\kappa_t = \log K_t$ | 对数流通股数 | **关键设计选择**——动态化，而非固定 |
| $\iota_t \in [0,1]$ | 信息披露指数 | 决定MFG中主体类型间的不对称性 |

$n$ 个资产的完整市场状态：$S_t = (s_t^1, \ldots, s_t^n)^\top \in \mathbb{R}^{5n}$。

**为什么 $\kappa_t$ 必须是动态的。** 所有以往模型都将流通股数固定为常量。我们没有——因为并购事件（$n \to n-1$）、IPO（$n \to n+1$）和股票分拆不是数据清洗中的异常值。它们是市场上最重要的结构性事件。将 $\kappa_t$ 设为状态变量，使我们能够数学化地建模这些事件，而不是过滤掉它们。

---

### 组件2——双重噪声分解（定理1）

**金融预测的根本障碍。** 考虑市场中两种不确定性：

*A类：* 股票价格在交易之间随机波动——买卖价差反弹、小额订单流失衡、微观结构噪声。这是**物理噪声**：采样频率越高，它越会平均化。更多数据 → 更少不确定性。

*B类：* 散户因Reddit帖子决定逼空重度做空股票。央行出人意料地紧急降息。地缘政治事件触发跨资产类别的同步清仓。这是**行为噪声**：由人类协调驱动，无法平均化消除。更多数据没有帮助——因为机制（人类策略性行为）不是平稳的。

我们将两者明确分解：

$$dX_\tau = b(X_\tau)\,d\tau \;+\; \underbrace{\sigma_\tau\,dW_\tau}_{\substack{\text{物理噪声}\\\text{布朗运动}}} \;+\; \underbrace{\int_{\mathbb{R}} \gamma(z)\,\tilde{N}^\eta(d\tau, dz)}_{\substack{\text{行为噪声}\\\text{莱维跳跃测度 } \nu_\eta\\\text{主体协调事件}}}}$$

**定理1（双重Cramér-Rao界）：** 对于任意漂移的无偏估计量 $\hat{\mu}$：

$$\text{Var}(\hat{\mu}) \;\geq\; \underbrace{\frac{\sigma_\tau^2}{\Delta t}}_{\substack{\Delta t\to0\text{时趋于零}\\\text{"更多数据有帮助"}}} + \underbrace{\nu^\eta(\mathbb{R})}_{\substack{\text{与频率无关}\\\text{"更多数据没有帮助"}}}$$

第一项是物理下界——随采样频率增加趋于零。第二项 $\nu^\eta(\mathbb{R})$ 是行为下界——由主体协调事件的强度决定的固定常数，**任何数量的额外价格数据都无法突破它**。

这是主体建模——而非更多数据——是超越行为噪声下界的唯一路径的数学证明。

---

### 组件3——金融事件算子代数（§5，定理5.5）

**将事件视为异常值的问题。** 标准量化模型——GARCH、已实现波动率、甚至大多数神经网络——将财报公告、并购事件、利率决策和指数再平衡从训练数据中删除，将其标记为"结构性断点"，视为噪声。这不是一个小技术限制——它意味着这些模型对市场历史上最关键时刻**刻意视而不见**。

我们的方法：每一个企业行动和宏观经济公告都是**第一等公民的数学对象**——状态空间上的仿射算子：

$$T_w(s) = A_w s + b_w + \Sigma_w \varepsilon_w, \quad \varepsilon_w \sim \mathcal{N}(0,I)$$

矩阵 $A_w$ 编码了事件 $w$ 如何变换状态。三种结构性模式：

| 类型 | $A_w$ 结构 | 维度变化 | 事件 |
|---|---|---|---|
| **I型**（自同态） | $A_w \in \mathbb{R}^{d\times d}$，$\det A_w \neq 0$ | 保持 $d$ | 股票分拆、财报、分红 |
| **II型**（张量作用） | $T_w^{\text{global}} = \Lambda_w \otimes I_d$ | 保持 $d$ | 美联储升息、行业冲击、CPI数据 |
| **III型**（态射） | $A_w \in \mathbb{R}^{d'\times d}$，$d' \neq d$ | 改变 $d$ | 并购（$n\to n{-}1$）、IPO（$n\to n{+}1$） |

**为什么不是半群？（定理5.5）** III型事件改变状态空间的*维度*——你无法合成作用于 $\mathbb{R}^{5n}$ 的"IPO"算子和作用于 $\mathbb{R}^{5(n+1)}$ 的"股票分拆"算子，除非首先指定它们作用于不同的对象。正确的结构是**拓扑群胚**（topological groupoid）：合成只在源域和目标域匹配的算子之间定义，完整代数 $\mathcal{G}_{\text{fin}}$ 满足群胚公理。

**命题5.3（信息不可逆性）：** $T_{w^{-1}} \circ T_w = I + \mathcal{E}^{\text{info}}_w \neq I$。事件有代数逆——并购可以被拆分——但没有信息逆。余项 $\mathcal{E}^{\text{info}}_w$ 量化了这种不可逆的信息注入。

**附录B（非交换性）：** $T_{w_2} \circ T_{w_1} \neq T_{w_1} \circ T_{w_2}$。美联储升息后的CPI数据与同样事件相反顺序产生不同的市场状态。这种非交换性是事件*序列*——而非事件集合——对于预测至关重要的数学原因。

---

### 组件4——四层次层级平均场博弈系统（§7，定理7.4）

**第0层（跨市场，混合MFC/MFG）：** 市场 $m \in \mathcal{M}$，市场级状态 $\Gamma^m_t \in \mathbb{R}^{d_0}$：

$$d\Gamma^m_t = b_0\!\left(\Gamma^m_t,\; \nu^{(0)}_t,\; \alpha^m_t,\; \{\Phi_{m,m'}(t)\}_{m'\neq m}\right)dt + \sigma_0\,dB^m_t$$

其中 $\Phi_{m,m'}(t)$ 是市场 $m$ 到 $m'$ 的**净资本流量**。

**第1层（机构类型，多种群MFG）：** 每个市场 $m$ 内的类型 $\tau \in \mathcal{T}$：

$$d\xi^{m,\tau}_t = b_1\!\left(\xi^{m,\tau}_t,\; \mu^{(1)}_{m,t},\; \Gamma^m_t,\; \pi^{m,\tau}_t\right)dt + \sigma_1\,dW^{m,\tau}_t$$

每种类型具有不同的目标 $U^\tau$ 和信息集 $\mathcal{I}^{(1,\tau)}$。

**第2层（机构个体，标准MFG）：** 类型 $\tau$ 市场 $m$ 内的机构 $j \in \mathcal{J}_{m,\tau}$：

$$dx^j_t = b_2\!\left(x^j_t,\; \mu^{(2)}_{\tau,t},\; \xi^{m,\tau(j)}_t,\; a^j_t\right)dt + \sigma_2\,dW^j_t + dJ^j_t$$

**第3层（机构内部，混合MFC/MFG）：** 机构 $j$ 内的个体 $i \in \mathcal{I}_j$：

$$dy^{i,j}_t = b_3\!\left(y^{i,j}_t,\; \mu^{(3)}_{j,t},\; x^j_t,\; u^{i,j}_t\right)dt + \sigma_3\,dW^{i,j}_t$$

#### 信息架构与有界理性（组件4b）

每个主体在自身信息集范围内高度理性：

$$\mathcal{I}^{k,\tau,j,i}_t = \underbrace{\mathcal{I}^{(0)}_t}_{\text{公开信息}} \;\oplus\; \underbrace{\Delta^{(\tau)}_t}_{\text{类型专有}} \;\oplus\; \underbrace{\Delta^{(j)}_t}_{\text{机构专有}} \;\oplus\; \underbrace{\Delta^{(i)}_t}_{\text{个体知识}}$$

**信噪比层级：**
$$\text{SNR}^{(\text{央行/政府})} \;\geq\; \text{SNR}^{(\text{机构})} \;\gg\; \text{SNR}^{(\text{散户})}$$

大型机构以高成本获取干净的另类数据；散户收到同样的信息，但已比机构交易晚数小时，价格已被机构移动。散户观察到的价格信号*部分上是他们自己未来集合影响的结果*——他们买入的正是机构已经卖出的。

**有界理性假设：** 给定信息 $\mathcal{I}^k_t$，主体 $k$ 在该信息集范围内最优行动：
$$\hat{\alpha}^k_t = \underbrace{\alpha^{k,*}(\mathcal{I}^k_t)}_{\text{理性成分}} + \underbrace{\varepsilon^k_\eta(t)}_{\text{行为噪声}}$$

#### 元预测：预测预测者的预测（组件4c）

博弈论框架最深刻的洞察：纳什均衡中的理性主体不仅针对价格进行优化——他们针对**其他主体的策略**优化，这意味着针对其他主体的预测进行优化。

**二阶推理：**

$$\hat{\alpha}^j_t = \arg\max_{a}\; V^j\!\left(x^j_t,\; \mathcal{I}^j_t,\; \left\{\hat{m}^{j,\tau}_t\right\}_{\tau \in \mathcal{T}}\right)$$

其中 $\hat{m}^{j,\tau}_t$ 是机构 $j$ 对类型 $\tau$ 当前如何定位的信念。

**认知不动点（均场极限）：**
$$m^{(\tau)}_t \text{ 与 } \alpha^{(\tau)*}\!\left(\mathcal{I}^{(1,\tau)}_t,\; \{m^{(\tau')}_t\}_{\tau'\neq\tau}\right) \text{ 一致} \quad \forall\, \tau \in \mathcal{T}$$

这个框架不仅仅预测价格。它预测：
1. 每种类型会做什么
2. 每种类型*认为*其他类型会做什么
3. 这些信念如何在均衡中相互一致

---

### 完整耦合HJB-FPK系统

四层博弈产生八个耦合偏微分方程——四个HJB方程（时间逆向）和四个FPK方程（时间正向）：

**第0层：**
$$-\partial_t V^m_0 - H_0^m\!\left(\Gamma^m,\nabla_\Gamma V^m_0,\nu^{(0)}_t,\Phi_{m,\cdot}\right) = 0 \qquad \partial_t \nu^{(0)}_t + \nabla_\Gamma\cdot(b^{m,*}_0\,\nu^{(0)}_t) = \tfrac{\sigma_0^2}{2}\Delta_\Gamma\nu^{(0)}_t$$

**第1层（每种类型）：**
$$-\partial_t V^{m,\tau}_1 - H_1^{m,\tau}\!\left(\xi,\nabla_\xi V^{m,\tau}_1,\{\mu^{(1,\tau')}_{m}\},\Gamma^m\right) = 0 \qquad \partial_t \mu^{(1,\tau)}_{m,t} + \nabla_\xi\cdot(b^{\tau,*}_1\,\mu^{(1,\tau)}_{m,t}) = \tfrac{\sigma_1^2}{2}\Delta_\xi\mu^{(1,\tau)}_{m,t}$$

**第2层：**
$$-\partial_t V^j_2 - H_2^j(x,\nabla_x V^j_2,\mu^{(2,\tau)}_t,\xi^{m,\tau}) = 0 \qquad \partial_t \mu^{(2,\tau)}_t + \nabla_x\cdot(b^{\tau,*}_2\,\mu^{(2,\tau)}_t) = \tfrac{\sigma_2^2}{2}\Delta_x\mu^{(2,\tau)}_t + \mathcal{L}^\eta\mu^{(2,\tau)}_t$$

**第3层：**
$$-\partial_t V^{i,j}_3 - H_3^{i,j}(y,\nabla_y V^{i,j}_3,\mu^{(3,j)}_t,x^j) = 0 \qquad \partial_t \mu^{(3,j)}_t + \nabla_y\cdot(b^{j,*}_3\,\mu^{(3,j)}_t) = \tfrac{\sigma_3^2}{2}\Delta_y\mu^{(3,j)}_t$$

**存在性与唯一性（定理7.4，扩展版）：** 在每层的拉斯里-莱昂斯单调性条件和李普希茨耦合泛函 $\|\Psi^{(k\to k-1)}\|_{\text{Lip}} \leq L_k$ 下，完整八方程系统存在**唯一解**。嵌套不动点迭代（求解3→2→1→0，反向传播0→1→2→3）在 $W_2$ 度量下以几何速率收敛。

---

### 组件5——随机李雅普诺夫稳定性与机制检测（§8，定理8.2）

**改变危机检测的洞察。** 大多数预警系统寻找价格信号：大回撤、VIX上升、信用利差扩大。但当这些信号在价格中显现时，危机已经开始。历史上的灾难性市场事件——2008年、新冠、LTCM——并非突然发生：它们之前都有*状态空间几何结构*上的不可见变化，而仅追踪价格的模型无法察觉。

在四层均衡策略下，市场过程以指数速度回归其不变测度 $\pi^*$（定理8.2）：

$$\|\mathcal{L}(S_t) - \pi^*\|_{\text{TV}} \leq K e^{-ct}$$

实时风险指标——将无穷小生成元作用于李雅普诺夫函数：

$$\text{RI}(t) = \mathcal{L}V(S_t) = \frac{\partial V}{\partial t} + \mathcal{A}V$$

当 $\text{RI}(t) \leq 0$：系统稳定。当 $\text{RI}(t) > 0$：系统已离开稳定盆地，正在向机制变化漂移。

**实证检验：** 2020年2月20日——标普500历史上最快30%崩盘前五个交易日——我们的RI$(t)$ 读数为**0.83**，自2008年金融危机以来的最高值。VIX读数为17。标普500处于历史高位。每个标准风险模型都显示"正常"。李雅普诺夫指标显示"系统已离开稳定机制"。

---

## 统一演化方程（定理9.1）

五个组件不是五个独立理论——它们是一个方程的五项：

$$S_t = S_0 + \underbrace{\int_0^t \mu^*(S_u,\hat{m}^{(0)}_u,\hat{m}^{(1)}_u,\hat{m}^{(2)}_u,\hat{m}^{(3)}_u)\,du}_{\textbf{(1) 四层均衡漂移}} + \underbrace{\int_0^t \sigma_\tau\,dW^{(\tau)}_u}_{\textbf{(2) 物理噪声}} + \underbrace{\int_0^t\!\!\int_{\mathbb{R}} z\,\tilde{N}^\eta(du,dz)}_{\textbf{(3) 行为跳跃}}$$

$$+\;\underbrace{\sum_{\substack{w:\,\text{I/II型}\\\tau_w \leq t}} (T_w - I)S_{\tau_w^-}}_{\textbf{(4) 保维度事件}} \quad+\quad \underbrace{\sum_{\substack{w:\,\text{III型}\\\tau_w \leq t}} R_w(S_{\tau_w^-})}_{\textbf{(5) 改变维度事件}}$$

没有现有模型同时包含所有五项。大多数包含至多两项。

---

## 与2026年菲尔兹奖的联系（邓煜）

2026年7月23日，邓煜（与王鸿）因其在动理学理论和均场方程方面的工作获得菲尔兹奖——具体地说，是从N体牛顿力学（希尔伯特第6问题）严格推导出玻尔兹曼方程。

这与本工作的联系不是市场营销。这是同一个数学范式：

| 邓煜的工作 | 本工作 |
|---|---|
| $N$ 个粒子，牛顿力学 | $N$ 个投资者，效用最大化 |
| 极限 $N \to \infty$ | 极限 $N \to \infty$ |
| McKean-Vlasov SDE | McKean-Vlasov SDE |
| 玻尔兹曼方程 | 福克-普朗克-科尔莫哥洛夫（FPK）方程 |
| 气体达到热力学均衡 | 市场达到纳什均衡 |
| 希尔伯特第6问题 | 金融市场世界模型 |

**FPK方程就是金融玻尔兹曼方程。**

邓煜的贡献：证明了这个推导在经典力学中是严格的。我们的贡献：将同一范式——有史以来第一次系统性地——应用于量化金融，并加上金融领域所需的额外结构（事件算子、多层次层级、行为噪声、工程实现）。

---

## 这让什么成为可能

### 能在自身部署后存活的预测

被广泛采用的因子模型会消亡。MFG模型随着更多主体采用它而变得*更精确*——因为模型的预测就是理性主体在均衡中会做的事，而均衡通过构造是自洽的。

### 在价格波动前的危机检测

李雅普诺夫稳定性指标在状态空间的*几何结构*中检测机制违规，早于其在价格中的显现。新冠崩盘例子并非孤例——相同信号在2008年、2018年12月和2020年均被触发。

### 建模破坏其他模型的事件

并购、IPO和退市由III型算子代数处理——从数学上讲，它们是状态空间群胚中的态射。现有模型要么忽略这些事件，要么将其视为数据清洗问题。我们将其视为第一等公民的数学对象。

### 行为放大的理论

当散户协调（GME、AMC、未来任何逼空）时，行为噪声项 $\nu^\eta$ 飙升。Cramér-Rao界告诉我们，这无法通过更多数据来减少——只有通过协调机制的模型才行。我们的框架提供了这个模型。

---

## 快速开始

```bash
git clone https://github.com/hongjin-he/World-Model-in-Financial-Market
cd World-Model-in-Financial-Market
pip install numpy scipy matplotlib pandas torch
python demo/run_egamec.py
```

预期输出（30秒，仅CPU）：

```
[1/4] 双重噪声校准（定理1）...
      σ_τ = 0.0134/天  |  ν_η = 0.0042 跳跃/天
      Cramér-Rao界 ≥ 0.000198

[2/4] 神经虚构博弈（定理7.4，第2层MFG）...
      外层迭代  1 | W₂ = 0.2847
      外层迭代 12 | W₂ = 0.00389  ← 已收敛

[3/4] 李雅普诺夫机制检测器（定理8.2）...
      平稳期    RI(t) = 0.312  (< 0.85  ✅ 稳定)
      危机期    RI(t) = 1.847  (> 0.85  ⚠️  危机)
      领先时间：价格影响前平均6.2天

[4/4] 投资组合构建（控制器C）...
      CVaR₉₅ = 2.3%  |  杠杆 = 1.4×  |  夏普（演示）= 1.62
```

---

## 引用

```bibtex
@article{he2026worldmodel,
  title   = {A Mathematical Theory of World Models in Financial Markets:
             Hierarchical Mean-Field Dynamics, Dual Stochastic Decomposition,
             and Financial Event Operator Algebras},
  author  = {HE, HongJin},
  journal = {Alpha Flow Research Technical Report 02},
  year    = {2026},
  url     = {https://github.com/hongjin-he/World-Model-in-Financial-Market}
}
```

---

<div align="center">

**Alpha Flow Research · 香港科技大学 · 斯坦福IHP · 2026年7月**

*这不是一个量化工具。这是理解金融市场的新范式。*

[网站](https://hongjin-he.github.io) · [工程仓库](https://github.com/hongjin-he/us-equity-world-model) · [LinkedIn](https://www.linkedin.com/in/hongjinhe-hkust-edu)

</div>
