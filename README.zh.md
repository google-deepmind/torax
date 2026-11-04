[![Unittests](https://github.com/google-deepmind/torax/actions/workflows/pytest.yml/badge.svg)](https://github.com/google-deepmind/torax/actions/workflows/pytest.yml)

<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

## 什么是 TORAX？

TORAX 是一个基于 JAX 的**可微托卡马克核心传输模拟器**（differentiable tokamak core transport simulator），专为快速且精确的正向建模、脉冲设计（pulse-design）、放电轨迹优化（trajectory optimization）和控制器设计工作流而设计。TORAX 采用 Python 和 JAX 编写，其核心设计初衷如下：

- **开源与可扩展性**：支持灵活的工作流耦合与模块集成。
- **JAX 自动微分与即时编译（JIT）带来极速运行表现**：可微性使得基于梯度的非线性偏微分方程（PDE）求解器能够实现快速而精确的建模，并能够对任意输入参数进行仿真结果的灵敏度分析（sensitivity analysis）。这赋能了放电轨迹优化以及半经验模型的数据驱动参数辨识等前沿应用。自动微分还使得这些应用能够通过添加新的物理模型（包括机器学习代理模型 ML-surrogates）或新的参数输入轻松扩展，无需手动推导雅可比矩阵（Jacobians）。
- **Python-JAX 是耦合物理模型机器学习代理（ML-surrogates）的天然框架**。

更详尽的文档请参阅我们的 [Read the Docs 官方文档页面](https://torax.readthedocs.io/)。

TORAX 具有以下物理与数值算法功能特性：

- **离子与电子热传输、电子粒子传输及电流扩散的耦合偏微分方程（Coupled PDEs）**：
    - 基于有限体积法（Finite-volume-method）的离散化。
    - 多种求解器选项：带 Pereverzev-Corrigan 项与预估-校正法（predictor-corrector method）的线性求解器、牛顿-拉夫逊（Newton-Raphson）非线性求解器，以及使用 `jaxopt` 库优化的非线性求解器。
    - 基于最后闭合磁力线面（LCFS）处总电流或环电压的极向磁通（Poloidal flux）边界条件。
- **欧姆加热功率、离子-电子热交换、聚变功率、韧致辐射（Bremsstrahlung）、杂质线辐射**，以及 [[ICRH 机器学习代理模型]](https://meetings.aps.org/Meeting/DPP24/Session/NP12.106)（目前覆盖有限区间）。
- **基于解析 Sauter 模型的新经典自举电流（bootstrap current）与电导率**。
- **耦合 [[QLKNN_7_11]](https://github.com/google-deepmind/fusion_surrogates/) 和 QLKNN10D [[van de Plassche et al, Phys. Plasmas 2020]](https://doi.org/10.1063/1.5134126) QuaLiKiz 神经网络代理模型**，用于基于第一性原理物理的湍流传输计算。
- **通用几何形貌支持**：通过 CHEASE、FBT 或 EQDSK 平衡态文件提供：
    - 出于测试与演示目的，`data/third_party/geo` 目录中提供了一个 CHEASE 平衡态文件。该文件对应于基于 [[Citrin et al, Nucl. Fusion 2010]](https://doi.org/10.1088/0029-5515/50/11/115007) 仿真的 ITER 杂化运行模式（hybrid scenario）平衡态，获取自 [PINT](https://gitlab.com/qualikiz-group/pyntegrated_model)。PINT 许可证文件可在 `data/third_party/geo` 中查阅。
    - 通过提供时间序列几何文件支持随时间变化的动态几何形貌。
- **简单的台基模型（Pedestal models）**：使用局部自适应源项设置内部边界条件。
- **锯齿振荡（Sawtooth）触发与剖面重分布**。

额外的辅助加热和电流驱动源可由用户自定义的解析模型或用户指定的预设数据提供。

模型实现已通过直接比对仿真输出与 RAPTOR [[Felici et al, Plasma Phys. Control. Fusion 2012]](https://iopscience.iop.org/article/10.1088/0741-3335/54/2/025002) 托卡马克传输模拟器进行了严格验证。

本仓库不是 Google 的官方正式支持产品（This is not an officially supported Google product）。

## 开发路线图 (Development roadmap)

完整的[开发路线图](https://torax.readthedocs.io/en/latest/roadmap.html#development-roadmap)已在我们的 Read the Docs 文档页面中列出。

## 安装指南 (Installation guide)

### 环境要求

安装 Python 3.12 或更高版本。

确保系统已安装 tkinter：

```shell
sudo apt-get install python3-tk
```

### 如何安装

#### 准备虚拟环境

安装 Virtualenv（若尚未安装）：

```shell
pip install --upgrade pip
```

```shell
pip install virtualenv
```

创建并激活虚拟环境：

```shell
python3 -m venv toraxvenv
source toraxvenv/bin/activate
```

#### 从 PyPI 安装

使用 TORAX 最简单的方式是通过 PyPI 进行安装：

```shell
pip install torax
```

您可以验证一切是否正常运行：

```shell
run_torax --config=examples/basic_config.py --quit
```

如果您计划参与 TORAX 的开发，则需要克隆该仓库，详见[贡献指南 (contributing)](https://torax.readthedocs.io/en/latest/contributing.html) 与[贡献贴士 (contribution_tips)](https://torax.readthedocs.io/en/latest/contribution_tips.html#contribution-tips)。

### 运行示例

以下命令将使用默认配置文件 `examples/basic_config.py` 运行 TORAX：

```shell
run_torax --config='examples/basic_config.py'
```

仿真进度将通过终端进度条显示，指示当前仿真时间和完成百分比。

若要运行更复杂的 ITER 启发式仿真，请运行：

```shell
run_torax --config='examples/iterhybrid_rampup.py'
```

以及：

```shell
run_torax --config='examples/iterhybrid_predictor_corrector.py'
```

额外的配置可以通过追加到上述运行命令的标志（flags）和环境变量来提供。例如，若需更详细的输出日志，可以使用 `--log_progress` 标志运行：

```shell
run_torax  --config='examples/iterhybrid_rampup.py' --log_progress
```

#### 设置环境变量

##### 错误检查 (Error checking)

若设置为 True，将在内部例程中启用错误检查，用于调试。默认为 False，因为该选项与持久化编译缓存不兼容。

```shell
$ export TORAX_ERRORS_ENABLED=<True/False>
```

##### JAX 编译与缓存 (JAX Compilation and Cache)

若设置为 False，JAX 将不对 TORAX 内部函数进行即时编译，用于调试。默认为 False。

```shell
$ export JAX_DISABLE_JIT=<True/False>
```

以下配置将启用 JAX 持久化缓存，使 JAX 将编译后的程序存储到文件系统中，从而在某些情况下减少重复编译时间：

```shell
$ export JAX_COMPILATION_CACHE_DIR=<自定义路径，如 ~/jax_cache>
$ export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=-1
$ export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0.0
```

#### 设置命令行标志

`log_progress` —— 输出仿真时间、步长 dt 以及每个时间步执行的求解器迭代次数（非线性求解器的 dt 回溯）。

```shell
run_torax \
   --config=examples/iterhybrid_predictor_corrector.py \
   --log_progress
```

`output_dir` —— 覆盖默认输出目录：

```shell
run_torax \
   --config=examples/iterhybrid_predictor_corrector.py \
   --output_dir=/path/to/output/dir
```

### 运行带完整断言检查的测试

若要运行启用了所有断言检查的 pytest 测试套件，可以使用以下命令：

```shell
TORAX_JAXTYPING=True TORAX_ERRORS_ENABLED=True pytest . -n auto
```

### 仿真后处理 (Post-simulation)

仿真完成后，仿真状态和衍生物理量的时间序列将被写入带有时间戳的文件，格式为 `state_history_%Y%m%d_%H%M%S.nc`。输出目录可由用户配置，默认为 `/tmp/torax_results`。

为了利用内存中（非持久化）缓存，进程在仿真结束时不会立即退出。您可以修改 runtime_params，切换 `log_progress` 和 `plot_progress` 标志，并重新运行仿真。以下修改示例将触发重新编译：

- 网格分辨率（Grid resolution）
- 演化变量（求解的方程）
- 更改使用的内部函数，例如传输模型、步长计算器（time_step_calculator）、源模型、求解器模型、台基模型

### 环境清理

您可以通过停用 Python 虚拟环境来退出：

```shell
deactivate
```

#### （可选）安装 QLKNN-hyper

TORAX 中的默认传输模型是 QLKNN_7_11，已作为 TORAX 依赖项安装。QLKNN_7_11 的一种替代方案是使用 QLKNN-hyper-10D，也称为 QLKNN10D [[K.L. van de Plassche PoP 2020]](https://doi.org/10.1063/1.5134126)。QLKNN_7_11 基于 QuaLiKiz 2.8.1，与 QLKNN10D 训练集相比具有改进的碰撞算符。QLKNN_7_11 训练数据将杂质密度梯度作为输入特征，并且与 QLKNN-hyper-10D 相比对靠近 LCFS 区域具有更好的覆盖度。然而，在特定用例中（例如与其他模拟器进行基准测试，或与 QLKNN_7_11 进行对比），使用 QLKNN10D 可能仍然很有价值。

下载 QLKNN-hyper 依赖项：

```shell
git clone https://gitlab.com/qualikiz-group/qlknn-hyper.git
```

若要使用 QLKNN10D，请将 TORAX 配置中 `transport` 部分的 `model_path` 设置为克隆仓库的绝对路径。

## 仿真教程 (Simulation tutorials)

请参阅我们 [Read the Docs 官方文档中的教程章节](https://torax.readthedocs.io/en/latest/tutorials.html)。

## 引用 TORAX (Citing TORAX)

TORAX 的论文已[发布在 arXiv 上](https://arxiv.org/abs/2406.06718)。如需引用 TORAX，请引用该论文：

```bibtex
@article{torax2024arxiv,
  title={{TORAX: A Fast and Differentiable Tokamak Transport Simulator in JAX}},
  author={Citrin, Jonathan and Goodfellow, Ian and Raju, Akhil and Chen, Jeremy
  and Degrave, Jonas and Donner, Craig and Felici, Federico and Hamel, Philippe
  and Huber, Andrea and Nikulin, Dmitry and Pfau, David and Tracey, Brendan, and
  Riedmiller, Martin and Kohli, Pushmeet},
  journal={arXiv preprint arXiv:2406.06718},
  year={2024}
}
```

---

> **翻译说明（Translation Disclaimer）**
> 本文档由社区志愿者（[@JasonYeYuhe](https://github.com/JasonYeYuhe)）人工翻译并维护，旨在降低中文开发者的阅读门槛。最后同步日期：2026年10月22日。若译文与英文原版存在任何出入或滞后，请以 [英文原版 (README.md)](README.md) 为准。欢迎通过 PR 协助改进译文！
