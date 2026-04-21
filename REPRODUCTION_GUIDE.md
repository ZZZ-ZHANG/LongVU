# LongVU 复现指南 - 新手友好版

> 适合刚接触 Python 和人工智能的研一学生，从零开始快速复现 LONGVU

## 📋 前提条件

- 已读懂论文
- 有 Linux 环境（或 WSL）
- 至少有 **一块 40GB+ 显存的 GPU**（如 A100/H100）
- 熟悉基本的命令行操作

---

## 🚀 第一步：环境搭建（约 30 分钟）

### 1.1 安装 Miniconda（如果没有）

```bash
# 下载并安装 Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# 按提示完成安装，然后重启终端
```

### 1.2 创建 Conda 环境

```bash
# 创建名为 longvu 的 Python 3.10 环境
conda create -n longvu python=3.10 -y
conda activate longvu
```

### 1.3 安装依赖

```bash
# 进入项目目录
cd /workspace

# 安装所有依赖（可能需要 15-20 分钟）
pip install -r requirements.txt
```

**常见问题：**
- 如果 `pip install` 失败，尝试使用清华镜像：
  ```bash
  pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
  ```

---

## 📥 第二步：下载预训练模型（约 20 分钟）

### 2.1 创建 checkpoints 目录

```bash
mkdir -p ./checkpoints
```

### 2.2 下载模型权重

选择其中一个模型（推荐 Qwen2_7B）：

**方案 A：LongVU_Qwen2_7B（视频理解能力强）**
```bash
cd ./checkpoints
git lfs install
git clone https://huggingface.co/Vision-CAIR/LongVU_Qwen2_7B
cd ..
```

**方案 B：LongVU_Llama3_2_3B（更轻量）**
```bash
cd ./checkpoints
git lfs install
git clone https://huggingface.co/Vision-CAIR/LongVU_Llama3_2_3B
cd ..
```

**注意：** 如果 HuggingFace 下载慢，可以使用镜像站或手动下载后上传到服务器。

---

## ✅ 第三步：快速测试推理（5 分钟）

### 3.1 运行测试脚本

我已经为你准备了新手友好的测试脚本：

```bash
python test_inference.py
```

**预期输出：**
```
============================================================
LongVU 推理测试 - 新手友好版
============================================================

[1/5] 正在加载模型...（首次加载较慢，请耐心等待）
✓ 模型加载成功！
   模型设备：cuda:0

[2/5] 正在加载测试视频...
✓ 视频加载成功！总帧数：300

[3/5] 正在抽取视频帧...
   视频 FPS: 30.00
   抽取帧数：10

[4/5] 正在预处理图像...
✓ 预处理完成！

[5/5] 正在生成描述...

============================================================
🎉 推理成功！模型输出：
============================================================
[模型生成的视频描述]
============================================================
```

### 3.2 如果报错怎么办？

**错误 1：找不到模型**
```
✗ 模型加载失败：...
请检查：1) checkpoints 目录下是否有模型文件 2) 路径是否正确
```
→ 确认 `./checkpoints/longvu_qwen` 文件夹存在且包含模型文件

**错误 2：CUDA out of memory**
```
RuntimeError: CUDA out of memory
```
→ 降低 batch size 或使用更小的模型（Llama3_2_3B）

**错误 3：缺少依赖**
```
ModuleNotFoundError: No module named 'xxx'
```
→ 重新运行 `pip install -r requirements.txt`

---

## 🔍 第四步：理解代码结构（重要！）

### 4.1 核心文件说明

```
/workspace/
├── longvu/                    # 核心代码
│   ├── builder.py            # 模型加载入口
│   ├── conversation.py       # 对话模板
│   ├── mm_datautils.py       # 数据处理工具
│   ├── multimodal_encoder/   # 视觉编码器
│   └── language_model/       # 语言模型
├── scripts/                   # 训练脚本
├── eval/                      # 评估脚本
├── examples/                  # 示例视频
└── test_inference.py         # 你的测试脚本 ⭐
```

### 4.2 关键代码流程

1. **加载模型** → `longvu/builder.py::load_pretrained_model()`
2. **读取视频** → `decord.VideoReader`
3. **抽帧处理** → `process_images()`
4. **构建 Prompt** → `conversation.py`
5. **生成回答** → `model.generate()`

建议逐行阅读 `test_inference.py`，理解每一步的作用。

---

## 🎯 第五步：尝试修改和实验

### 5.1 修改输入问题

编辑 `test_inference.py` 第 73 行：
```python
qs = "Describe this video in detail"  # 改成你想问的问题
```

例如：
```python
qs = "What is the main activity in this video?"
qs = "How many people are in the video?"
qs = "用中文描述这个视频"  # 试试中文！
```

### 5.2 更换测试视频

编辑 `test_inference.py` 第 43 行：
```python
video_path = "./examples/video1.mp4"  # 换成你的视频路径
```

你可以把自己的视频放到 `examples/` 目录下。

### 5.3 调整生成长度

编辑 `test_inference.py` 第 93 行：
```python
max_new_tokens=256,  # 增大可以生成更长的回答
```

---

## 📚 第六步：深入学习（可选）

### 6.1 阅读官方文档

- [推理代码详解](docs/inference.md)
- [评估方法](docs/eval.md)
- [模型架构](docs/models.md)

### 6.2 尝试训练自己的模型

**警告：** 训练需要大量 GPU 资源（64 块 H100），不建议新手立即尝试。

如果一定要试：
```bash
# 1. 准备数据集（参考 README.md 的 Dataset 部分）
# 2. 修改脚本中的路径
# 3. 运行训练脚本
sh scripts/train_image_qwen.sh
```

### 6.3 参与开源社区

- 查看 GitHub Issues：https://github.com/Vision-CAIR/LongVU/issues
- 加入讨论群（如果有）
- 提交你的改进或 Bug 报告

---

## 🆘 常见问题 FAQ

### Q1: 我没有 40GB GPU 怎么办？
**A:** 可以尝试：
1. 使用更小的模型（Llama3_2_3B）
2. 使用量化版本（需要自己转换）
3. 使用云 GPU 服务（如 Colab Pro、AutoDL 等）

### Q2: 下载模型太慢怎么办？
**A:** 
1. 使用镜像站
2. 手动下载后上传到服务器
3. 找同学拷贝

### Q3: 我想用 Windows 怎么办？
**A:** 参考 README.md 的 Windows 部分，但强烈建议使用 Linux 或 WSL。

### Q4: 代码看不懂怎么办？
**A:** 
1. 先跑通测试脚本
2. 逐行阅读代码，配合 print 调试
3. 查阅 PyTorch 和 Transformers 文档
4. 问 ChatGPT 或 Claude

---

## 📝 学习路线建议

```
Week 1: 环境搭建 + 跑通推理 ✓
Week 2: 理解代码结构 + 修改参数实验
Week 3: 深入阅读论文 + 对比代码实现
Week 4: 尝试微调或改进模型
```

---

## 🎉 恭喜你！

如果你已经成功运行了测试脚本，说明你已经：
- ✅ 搭建了完整的 AI 开发环境
- ✅ 理解了多模态模型的基本推理流程
- ✅ 迈出了复现论文的第一步

接下来可以继续探索更多功能，或者开始研究如何改进这个方法！

**加油！科研之路从此开始 🚀**
