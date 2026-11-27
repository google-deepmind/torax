# TORAX

[![Unittests][pytest-badge]][pytest-workflow]
[![Documentation][docs-badge]][docs]

[ [English](https://github.com/google-deepmind/torax/blob/main/README.md) | 简体中文 ]

TORAX 是一个使用 Python 和 [JAX][jax] 编写的可微一维（1D）托卡马克芯部等离子体输运模拟器，专为快速正向建模、脉冲设计、轨迹优化与控制器设计打造。其核心设计原则包括：

- **模块化与可扩展**：易于为输运、源项、新经典效应等注册自定义模型。
- **快速与自动可微**：支持在 CPU 与 GPU 上进行端到端 `jax.jit` 编译，无需手推雅可比矩阵即可运行基于梯度的非线性偏微分方程（PDE）求解器，并原生支持通过 `jax.grad` / `jax.vmap` 进行灵敏度分析与参数优化。
- **原生契合机器学习代理模型（ML-surrogate）**：通过 JAX 无缝耦合神经网络物理代理模型，包括来自 [`fusion_surrogates`][fusion-surrogates] 的湍流输运模型。

完整文档与物理细节请参阅 [torax.readthedocs.io][docs]。正在从 TORAX v1 升级？请参阅 [v1 至 v2 迁移指南][migration-guide]。

## 物理模型与数值方法

- **耦合的一维输运偏微分方程（PDEs）**
  - 采用有限体积法（Finite-volume）对离子热输运、电子热输运、电子粒子输运以及极向磁通扩散进行离散化，支持多种边界条件。
  - 支持均匀或非均匀网格。
  - 提供多种求解器选项，包括预估-校正法（predictor-corrector）、牛顿-拉夫逊法（Newton-Raphson）以及基于 JAX 的优化器。
- **湍流与新经典输运**
  - 支持多组分芯部与台基（pedestal）输运，涵盖机器学习代理模型（QLKNN、TGLFNN）、经验模型（Bohm/gyro-Bohm）以及高保真准线性模型（QuaLiKiz、TGLF）。
  - 提供针对自举电流（bootstrap current）、电导率、新经典输运及极向速度的快速解析模型。
- **台基、边界与磁流体动力学（MHD）**
  - 模块化台基结构，其形成由 L-H 功率阈值定标律触发，支持作为边界条件强制施加或通过输运方程演化。
  - 内置与扩展 Lengyel 一维刮削层/偏滤器（SOL/divertor）模型的耦合。
  - 简化的锯齿崩塌（sawtooth crash）模型。

- **源项、汇项与平衡态**
  - 涵盖聚变反应、杂质辐射与韧致辐射（Bremsstrahlung）等芯部物理过程的解析模型。
  - 针对弹丸注入（pellets）、离子与电子回旋加热及电流驱动等执行器的基础模型；粒子源配备反馈控制器以匹配实验密度剖面。
  - 支持来自 CHEASE、MEQ、EQDSK、IMAS 与 TokaMaker 的静态及时变平衡态。

## 安装指南

TORAX 需要 Python 3.12 或更高版本。建议在虚拟环境中从 PyPI 安装最新稳定版：

```shell
python3 -m venv toraxvenv
source toraxvenv/bin/activate
pip install torax
```

如需进行开发（包含测试与代码检查依赖的可编辑安装）：

```shell
git clone https://github.com/google-deepmind/torax.git
cd torax
pip install -e ".[dev]"
```

有关 GPU 环境配置以及可选的 `QLKNN10D` 或 `tglf2py` 扩展安装说明，请参阅[安装指南][installation]。

## 快速上手

### 命令行工具

从终端运行示例配置：

```shell
run_torax --config=examples/iterhybrid_rampup.py --output_dir=/tmp/torax_results
```

使用交互式 Plotly 图表查看一个或多个已保存的运行结果：

```shell
plot_torax --outfile /tmp/torax_results/state_history_<timestamp>.nc
```

### Python API

```python
import torax

torax_config = torax.build_torax_config_from_file(
    'examples/iterhybrid_rampup.py'
)  # 或使用 torax.ToraxConfig.from_dict(config_dict)

state_history = torax.run_simulation(torax_config)

data_tree = state_history.simulation_output_to_xr()
data_tree.to_netcdf('state_history.nc')
```

可在 Python 中直接使用 `torax.plot_run_from_data_tree`（内存数据）或 `torax.plot_run`（读取 `.nc` 文件）绘制运行结果。更多详情请参阅[以编程方式运行][programmatic]。

### JAX 缓存、调试与测试

如需启用持久化 JAX 编译缓存，请运行：

```shell
export JAX_COMPILATION_CACHE_DIR=~/jax_cache
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=-1
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0.0
```

调试时，可设置 `JAX_DISABLE_JIT=True` 禁用 JIT 编译，或设置 `TORAX_ERRORS_ENABLED=True` 启用内部运行时检查。在源码检出目录中运行单元测试套件：

```shell
TORAX_JAXTYPING=True TORAX_ERRORS_ENABLED=True pytest -n auto
```

## 引用 TORAX

如果您在研究中使用了 TORAX，请引用 [arXiv 上的 TORAX 论文][paper]：

```
@article{torax2024arxiv,
  title={{TORAX: A Fast and Differentiable Tokamak Transport Simulator in JAX}},
  author={Citrin, Jonathan and Goodfellow, Ian and Raju, Akhil and Chen, Jeremy
  and Degrave, Jonas and Donner, Craig and Felici, Federico and Hamel, Philippe
  and Huber, Andrea and Nikulin, Dmitry and Pfau, David and Tracey, Brendan and
  Riedmiller, Martin and Kohli, Pushmeet},
  journal={arXiv preprint arXiv:2406.06718},
  year={2024}
}
```

有兴趣参与贡献？请参阅[贡献指南][contributing]文档。

本项目并非 Google 官方支持的产品。

[pytest-badge]: https://github.com/google-deepmind/torax/actions/workflows/pytest.yml/badge.svg
[pytest-workflow]: https://github.com/google-deepmind/torax/actions/workflows/pytest.yml
[docs-badge]: https://img.shields.io/readthedocs/torax/latest
[docs]: https://torax.readthedocs.io/
[jax]: https://docs.jax.dev/
[fusion-surrogates]: https://github.com/google-deepmind/fusion_surrogates
[migration-guide]: https://torax.readthedocs.io/en/latest/migration_guide.html
[installation]: https://torax.readthedocs.io/en/latest/installation.html
[programmatic]: https://torax.readthedocs.io/en/latest/running_programmatically.html
[contributing]: https://torax.readthedocs.io/en/latest/contributing.html
[paper]: https://arxiv.org/abs/2406.06718

---

> _注：本中文文档由社区志愿者翻译与维护，旨在降低中文开发者的阅读门槛。如与英文原版存在差异或更新延迟，请以英文原版 [README.md](https://github.com/google-deepmind/torax/blob/main/README.md) 为唯一权威标准（Single Source of Truth）。_
