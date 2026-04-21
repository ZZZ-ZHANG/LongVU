#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔍 LongVU 代码结构探索器 - 专为新手设计

这个脚本会帮你：
1. 画出项目目录树
2. 标注哪些文件重要（⭐️核心 / 🟡次要 / ⚪可忽略）
3. 指出阅读顺序
4. 解释每个模块的作用

运行方式：python explore_code_structure.py
"""

import os
from pathlib import Path

# 颜色定义
COLORS = {
    'red': '\033[91m',
    'green': '\033[92m',
    'yellow': '\033[93m',
    'blue': '\033[94m',
    'purple': '\033[95m',
    'cyan': '\033[96m',
    'end': '\033[0m',
    'bold': '\033[1m'
}

def colorize(text, color):
    return f"{COLORS.get(color, '')}{text}{COLORS['end']}"

# 文件重要性标注
# ⭐️ = 必须看 (核心逻辑)
# 🟡 = 建议看 (辅助理解)
# ⚪ = 暂时忽略 (工程化代码)
FILE_IMPORTANCE = {
    # 入口文件
    'test_inference.py': ('⭐️', '推理入口 - 从这里开始！看数据怎么流进模型'),
    'app.py': ('🟡', 'Gradio 界面入口 - 想了解 UI 可以看'),
    
    # 核心模型架构
    'cambrian_arch.py': ('⭐️', '模型核心架构 - forward 函数在这里，论文实现的关键'),
    'builder.py': ('⭐️', '模型工厂 - 看如何组装各个组件'),
    
    # 多模态编码器
    'siglip_encoder.py': ('🟡', '视觉编码器 - 视频帧怎么变成特征向量'),
    'base_encoder.py': ('⚪', '编码器基类 - 暂时不用看'),
    
    # 数据处理
    'conversation.py': ('⭐️', '对话模板 - 看问题和视频怎么拼成输入'),
    'mm_datautils.py': ('🟡', '多模态数据处理 - 训练时用'),
    
    # 配置文件
    'constants.py': ('🟡', '常量定义 - 路径、特殊 token 等'),
    
    # 评测脚本
    'eval_mvbench.py': ('⚪', 'MVBench 评测 - 复现成功后再看'),
    'eval_videomme.py': ('⚪', 'Video-MME 评测 - 暂时忽略'),
    
    # 工具函数
    'utils.py': ('⚪', '通用工具 - 用到时再查'),
    '__init__.py': ('⚪', '包初始化 - 机器看的，人不用管'),
}

# 目录说明
DIR_DESCRIPTIONS = {
    'longvu': '📦 核心代码包 - 90% 的时间在这里',
    'longvu/multimodal_encoder': '👁️ 视觉编码器 - 把视频/图片变成机器能懂的特征',
    'longvu/language_model': '🗣️ 语言模型 - Qwen2 的包装，负责生成文字',
    'longvu/multimodal_projector': '🔗 投影层 - 把视觉特征映射到语言空间',
    'eval': '📊 评测脚本 - 在标准数据集上测试性能',
    'scripts': '📜 Shell 脚本 - 一键运行训练/推理的命令',
    'checkpoints': '💾 模型权重 - 下载预训练模型放这里',
    'data': '📁 数据集 - 训练/测试用的视频数据',
    'docs': '📖 文档 - 可能有详细说明',
}

def get_importance_marker(filename):
    """获取文件重要性标记"""
    if filename in FILE_IMPORTANCE:
        return FILE_IMPORTANCE[filename]
    
    # 默认规则
    if filename.endswith('__init__.py'):
        return ('⚪', '包初始化文件')
    elif 'arch' in filename or 'model' in filename:
        return ('⭐️', '可能是模型架构核心')
    elif 'builder' in filename:
        return ('⭐️', '组件构建器')
    elif 'config' in filename:
        return ('🟡', '配置文件')
    elif 'eval' in filename:
        return ('⚪', '评测脚本')
    elif 'utils' in filename or 'helper' in filename:
        return ('⚪', '工具函数')
    else:
        return ('🟡', '普通 Python 文件')

def print_tree(start_path='.', prefix='', is_last=True, depth=0, max_depth=3):
    """打印目录树，带重要性标注"""
    start_path = Path(start_path)
    
    if depth > max_depth:
        return
    
    # 获取当前目录下的所有项
    try:
        items = sorted([p for p in start_path.iterdir() 
                       if not p.name.startswith('.') and p.name != '__pycache__'])
    except PermissionError:
        return
    
    # 过滤：只显示 .py 文件和目录
    items = [p for p in items if p.is_dir() or p.suffix == '.py']
    
    if not items:
        return
    
    for i, item in enumerate(items):
        is_last_item = (i == len(items) - 1)
        connector = "└── " if is_last_item else "├── "
        
        # 构建显示名称
        display_name = item.name
        
        # 添加图标和说明
        extra_info = ""
        if item.is_file():
            marker, description = get_importance_marker(item.name)
            if marker == '⭐️':
                display_name = colorize(display_name, 'green')
                extra_info = f" {marker} {colorize(description, 'cyan')}"
            elif marker == '🟡':
                display_name = colorize(display_name, 'yellow')
                extra_info = f" {marker} {colorize(description, 'blue')}"
            else:
                display_name = colorize(display_name, 'white')
                extra_info = f" {marker} {colorize(description, 'gray')}"
        
        elif item.is_dir():
            display_name = colorize(display_name, 'purple')
            if item.name in DIR_DESCRIPTIONS:
                extra_info = f" 📂 {colorize(DIR_DESCRIPTIONS[item.name], 'blue')}"
        
        # 打印当前行
        print(f"{prefix}{connector}{display_name}{extra_info}")
        
        # 递归打印子目录
        if item.is_dir():
            extension = "    " if is_last_item else "│   "
            print_tree(item, prefix + extension, is_last_item, depth + 1, max_depth)

def print_reading_guide():
    """打印阅读指南"""
    print("\n" + "="*80)
    print(colorize("📚 LongVU 代码阅读路线图", 'bold'))
    print("="*80)
    
    guide = """
【第 1 步】从入口开始 (30 分钟)
   └─ 打开 test_inference.py
   └─ 搞清楚：输入是什么？输出是什么？中间调了哪些函数？
   
【第 2 步】理解数据流 (1 小时)
   └─ 追踪数据：视频文件 -> 帧 -> 特征向量 -> 模型输入
   └─ 重点看：conversation.py (怎么拼接问题+视频)
   
【第 3 步】深入模型核心 (2-3 小时)
   └─ 打开 cambrian_arch.py (或 longvu_arch.py)
   └─ 找到 forward() 函数
   └─ 对照论文 Figure 2，找时空压缩的代码实现
   
【第 4 步】理解组件组装 (1 小时)
   └─ 打开 multimodal_encoder/builder.py
   └─ 看 SigLIP/DINO 编码器是怎么被加载的
   
【第 5 步】选择性深入 (按需)
   └─ 想改模型结构？→ 深入研究 cambrian_arch.py
   └─ 想用自己的数据？→ 研究 mm_datautils.py
   └─ 想做评测？→ 看 eval/ 目录下的脚本
"""
    
    for line in guide.split('\n'):
        if line.strip().startswith('【'):
            print(colorize(line, 'green'))
        elif line.strip().startswith('└─'):
            print(colorize(line, 'yellow'))
        elif line.strip():
            print(line)
    
    print("\n" + "="*80)
    print(colorize("💡 新手提示", 'bold'))
    print("="*80)
    print("""
1. 不要按字母顺序读文件！按上面的路线图来。
2. 遇到看不懂的函数，先假设它"能工作"，继续往下读。
3. 用 VS Code 的"跳转到定义"功能 (F12)，但不要追太深。
4. 在关键地方 print() 输出变量形状，比干看代码管用。
5. 这个项目的核心只有 3-5 个文件，其他都是辅助。
""")

def print_growth_process():
    """解释项目是如何从零长大的"""
    print("\n" + "="*80)
    print(colorize("🌱 这个项目是怎么从零变成这样的？", 'bold'))
    print("="*80)
    
    process = """
阶段 1: 灵光一现 (作者的第一周)
   只有一个文件：try_longvu.py
   - 硬编码路径
   - 没有错误处理
   - 能跑就行
   
阶段 2: 功能拆分 (作者的第二周)
   发现代码太乱，开始拆分：
   - 模型结构 -> cambrian_arch.py
   - 数据加载 -> mm_datautils.py
   - 主逻辑 -> train.py / inference.py
   
阶段 3: 工程化 (作者的第一个月)
   为了发论文、支持更多功能：
   - 加配置系统 -> config/
   - 支持多种编码器 -> multimodal_encoder/
   - 加评测脚本 -> eval/
   - 写安装脚本 -> setup.py
   
你现在看到的是"装修好的房子"，但复现时要回到"毛坯房"状态。
你的任务：找到那个"最初的 try_longvu.py"精神内核。
"""
    
    for line in process.split('\n'):
        if '阶段' in line:
            print(colorize(line, 'purple'))
        elif line.strip().startswith('-'):
            print(colorize(line, 'yellow'))
        elif line.strip():
            print(line)

if __name__ == '__main__':
    print("\n" + "="*80)
    print(colorize("🔍 LongVU 代码结构探索器", 'bold'))
    print(colorize("为你的复现之旅绘制地图", 'cyan'))
    print("="*80)
    
    print("\n📁 项目目录结构 (⭐️=核心必看 | 🟡=建议看 | ⚪=暂时忽略):\n")
    print_tree('.', max_depth=3)
    
    print_reading_guide()
    print_growth_process()
    
    print("\n" + "="*80)
    print(colorize("🚀 下一步行动", 'bold'))
    print("="*80)
    print("""
1. 打开终端，运行：code test_inference.py (或用你喜欢的编辑器)
2. 从头到尾读一遍，不懂的地方打个问号，不要停
3. 读完后来问我："test_inference.py 里的 XXX 是什么意思？"

加油！你已经迈出了最重要的一步：知道从哪里开始。
""")
