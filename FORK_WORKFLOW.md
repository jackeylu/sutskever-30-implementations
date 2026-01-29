# Fork 工作流设置指南

本指南帮助你设置 fork 工作流，既能提交到自己的仓库，又能方便地获取原仓库的更新。

## 📋 工作流概述

```
原仓库 (pageman/sutskever-30-implementations)
    ↓
  你 fork
    ↓
你的仓库 (your-username/sutskever-30-implementations) ← origin
    ↓
  你推送修改
    ↓
从原仓库获取更新 → 合并到你的分支
```

## 步骤 1: 在 GitHub 上 Fork 原仓库

1. 访问原仓库：https://github.com/pageman/sutskever-30-implementations
2. 点击右上角的 **"Fork"** 按钮
3. 选择你的 GitHub 账号作为目标
4. 等待 fork 完成（通常几秒钟）

## 步骤 2: 配置 Git 远程仓库

Fork 完成后，在项目目录执行以下命令：

### 2.1 添加你自己的 fork 作为 origin

```bash
# 替换 YOUR_USERNAME 为你的 GitHub 用户名
git remote set-url origin git@github.com:YOUR_USERNAME/sutskever-30-implementations.git
```

### 2.2 添加原仓库作为 upstream

```bash
git remote add upstream git@github.com:pageman/sutskever-30-implementations.git
```

### 2.3 验证远程仓库配置

```bash
git remote -v
```

应该看到类似输出：
```
origin    git@github.com:YOUR_USERNAME/sutskever-30-implementations.git (fetch)
origin    git@github.com:YOUR_USERNAME/sutskever-30-implementations.git (push)
upstream  git@github.com:pageman/sutskever-30-implementations.git (fetch)
upstream  git@github.com:pageman/sutskever-30-implementations.git (push)
```

## 步骤 3: 提交你的修改

### 3.1 查看当前状态

```bash
git status
```

### 3.2 添加并提交修改

```bash
# 添加所有修改
git add .

# 或选择性添加
git add requirements.txt
git add VIRTUALENV_SETUP.md
git add CLAUDE.md

# 提交修改
git commit -m "feat: Add Python virtual environment setup and documentation

- Add requirements.txt with all dependencies
- Add VIRTUALENV_SETUP.md with detailed setup guide
- Add CLAUDE.md for Claude Code guidance
- Create venv/ for local development (gitignored)"
```

### 3.3 推送到你的 fork

```bash
# 当前在 learn 分支，推送到你的仓库
git push -u origin learn
```

## 步骤 4: 获取原仓库更新

### 定期同步上游更新（建议每周执行一次）

#### 方式 1: 使用 merge（推荐，保留完整历史）

```bash
# 1. 确保在 main 分支
git checkout main

# 2. 从 upstream 获取最新代码
git fetch upstream

# 3. 合并 upstream/main 到本地 main
git merge upstream/main

# 4. 推送更新到你的 fork（可选，保持你的 fork 与原仓库同步）
git push origin main

# 5. 如果需要，将更新合并到你的工作分支
git checkout learn
git merge main
```

#### 方式 2: 使用 rebase（保持线性历史）

```bash
# 1. 确保在 main 分支
git checkout main

# 2. 从 upstream 获取最新代码
git fetch upstream

# 3. Rebase 你的 main 到 upstream/main
git rebase upstream/main

# 4. 推送更新（可能需要 force push）
git push origin main --force-with-lease

# 5. Rebase 你的工作分支
git checkout learn
git rebase main
```

## 步骤 5: 处理合并冲突

如果在合并上游更新时遇到冲突：

```bash
# 1. Git 会提示有冲突的文件
# 2. 打开冲突文件，查找 <<<<<<<, =======, >>>>>>> 标记
# 3. 手动解决冲突
# 4. 标记冲突已解决
git add <resolved-file>

# 5. 继续合并
git commit  # 对于 merge
# 或
git rebase --continue  # 对于 rebase
```

## 📅 常用工作流场景

### 场景 1: 日常开发

```bash
# 1. 在工作分支上工作
git checkout learn

# 2. 添加修改并提交
git add .
git commit -m "your commit message"

# 3. 推送到你的 fork
git push origin learn
```

### 场景 2: 获取上游最新更新

```bash
# 1. 切换到 main
git checkout main

# 2. 拉取上游更新
git fetch upstream
git merge upstream/main

# 3. 推送到你的 fork
git push origin main

# 4. 合并到你的工作分支
git checkout learn
git merge main
```

### 场景 3: 从上游 main 创建新分支

```bash
# 1. 确保主分支是最新的
git checkout main
git fetch upstream
git merge upstream/main

# 2. 从 main 创建新分支
git checkout -b feature/new-feature

# 3. 在新分支上工作
git add .
git commit -m "Add new feature"

# 4. 推送到你的 fork
git push -u origin feature/new-feature
```

## 🔧 实用命令

### 查看远程仓库
```bash
git remote -v
```

### 查看所有分支
```bash
git branch -a
```

### 查看上游更新（不合并）
```bash
git fetch upstream
git log HEAD..upstream/main  # 查看上游的新提交
git diff HEAD upstream/main  # 查看差异
```

### 同步你的 fork 到上游 main 的状态
```bash
git checkout main
git fetch upstream
git reset --hard upstream/main
git push origin main --force
```

## 📝 提交信息规范

建议使用清晰的提交信息格式：

```
<type>: <subject>

<body>

<footer>
```

**类型（type）**:
- `feat`: 新功能
- `fix`: 修复 bug
- `docs`: 文档更新
- `style`: 代码格式调整
- `refactor`: 代码重构
- `test`: 测试相关
- `chore`: 构建或辅助工具的变动

**示例**:
```
feat: Add Python virtual environment setup

- Create requirements.txt with core dependencies
- Add VIRTUALENV_SETUP.md with detailed guide
- Add CLAUDE.md for repository guidance
```

## ⚠️ 注意事项

1. **永远不要直接推送到 upstream**（你也没有权限）
2. **定期同步上游更新**，避免你的 fork 过于落后
3. **解决冲突时要小心**，确保不删除原仓库的重要更新
4. **使用有意义的分支名称**，便于管理
5. **提交前先 pull**，减少冲突可能性

## 🎯 最佳实践

1. **为每个新功能/修复创建独立分支**
2. **保持 main 分支与 upstream 同步**
3. **使用清晰的提交信息**
4. **定期推送到你的 fork**，作为备份
5. **在合并上游更新前先查看日志**，了解更新内容

## 📚 相关资源

- [GitHub Fork 文档](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks)
- [Git 远程仓库管理](https://git-scm.com/book/en/v2/Git-Basics-Working-with-Remotes)
- [合并冲突解决](https://git-scm.com/book/en/v2/Git-Tools-Advanced-Merging)

---

**设置完成后，你就可以自由地提交修改到你的仓库，同时随时获取原仓库的更新！** 🚀
