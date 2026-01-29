# Python 虚拟环境设置指南

这个虚拟环境已经配置好，用于学习、改进和记录 Sutskever 30 篇论文的实现。

## 虚拟环境位置
```
venv/
```

## 激活虚拟环境

### Windows (Git Bash / MSYS2)
```bash
source venv/Scripts/activate
```

### Windows (CMD)
```cmd
venv\Scripts\activate.bat
```

### Windows (PowerShell)
```powershell
venv\Scripts\Activate.ps1
```

### Linux / macOS
```bash
source venv/bin/activate
```

激活成功后，命令行提示符会显示 `(venv)` 前缀。

## 已安装的依赖

- **numpy** (≥1.21.0) - 数值计算核心库
- **matplotlib** (≥3.4.0) - 数据可视化
- **scipy** (≥1.7.0) - 科学计算
- **jupyter** (≥1.0.0) - Jupyter 核心工具
- **notebook** (≥6.4.0) - Jupyter Notebook 界面
- **ipykernel** (≥6.0.0) - Python 内核 for Jupyter
- **ipywidgets** (≥7.6.0) - 交互式组件

## 使用指南

### 1. 启动 Jupyter Notebook
```bash
# 激活虚拟环境后
jupyter notebook
```

### 2. 运行特定的 notebook
```bash
jupyter notebook 02_char_rnn_karpathy.ipynb
```

### 3. 检查已安装的包
```bash
pip list
```

### 4. 安装额外的包（如果需要）
```bash
pip install package_name
```

### 5. 更新 requirements.txt（如果添加了新依赖）
```bash
pip freeze > requirements.txt
```

## 退出虚拟环境

```bash
deactivate
```

## 常见任务

### 学习一个新的论文实现
1. 激活虚拟环境
2. 运行 `jupyter notebook`
3. 选择对应的编号 notebook（例如 `02_char_rnn_karpathy.ipynb`）
4. 从上到下执行所有单元格
5. 阅读每个部分的解释和可视化

### 改进现有实现
1. 在虚拟环境中打开 notebook
2. 修改代码单元格
3. 使用 "Kernel → Restart & Run All" 测试修改
4. 保存更改

### 创建新的笔记或记录
1. 使用 Jupyter 创建新的 notebook
2. 或编辑现有的 `.md` 文件（如 README.md、IMPLEMENTATION_TRACKS.md）

## 验证安装

运行以下命令验证环境设置正确：

```bash
# 激活虚拟环境后
python -c "import numpy; import matplotlib; import scipy; print('✓ 所有核心包已安装')"
python -c "import jupyter; print('✓ Jupyter 已安装')"
```

## 目录结构说明

```
sutskever-30-implementations/
├── venv/                          # ← 虚拟环境目录（新创建）
├── 01_complexity_dynamics.ipynb
├── 02_char_rnn_karpathy.ipynb
├── ...
├── 30_lost_in_middle.ipynb
├── requirements.txt               # ← 依赖列表（新创建）
├── README.md
├── IMPLEMENTATION_TRACKS.md
└── VIRTUALENV_SETUP.md           # ← 本文件
```

## 注意事项

1. **始终在激活虚拟环境后工作**，确保使用的是正确的 Python 和包
2. **不要提交 venv/ 到 Git**（已在 .gitignore 中排除）
3. **requirements.txt** 已提交到版本控制，方便他人复现环境
4. 如果需要删除虚拟环境，直接删除 `venv/` 目录即可，然后重新创建

## Python 版本
当前系统 Python 版本: **Python 3.14.2**

虚拟环境使用系统的 Python 版本创建。

---

**准备好开始学习了！运行 `jupyter notebook` 开始探索这 30 篇经典论文的实现吧！** 🚀
