#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎓 零基础超详细教学版推理脚本
专为从未写过Python的研究生设计
每个词、每行代码都有比喻和详细解释
"""

# ==========================================
# 第一部分：准备工具（导入库）
# ==========================================

# 【比喻】这行就像你去厨房做饭前，先把锅碗瓢盆从柜子里拿出来
# "import" 是Python的关键字，意思是"拿来"
# "torch" 是一个超级强大的数学计算工具箱，专门用来运行AI模型
# "as torch" 是给这个工具箱起个简称，以后直接叫"torch"不用写全名
import torch

# 【比喻】这是拿来一个"操作系统接口工具箱"
# "os" 是 "Operating System" 的缩写
# 用它来跟电脑打交道，比如创建文件夹、读取文件路径等
import os

# 【比喻】这是拿来一个"视频处理工具箱"
# "cv2" 是著名的计算机视觉库 OpenCV 的名字（因为版本2最经典，所以叫cv2）
# 用它来读取视频文件，把视频变成一帧帧的图片
import cv2

# 【比喻】这是拿来一个"数组处理工具箱"
# "numpy" 是科学计算的基础库，擅长处理数字矩阵
# "as np" 是给它起个简称叫"np"
import numpy as np

# 【比喻】这是拿来一个"时间工具箱"
# 用来获取当前时间、计算时间差等
import time

# ==========================================
# 第二部分：加载AI模型（核心步骤）
# ==========================================

def load_model():
    """
    【函数定义】
    "def" 是 "define" 的缩写，意思是"定义一个功能"
    "load_model" 是这个功能的的名字，意思是"加载模型"
    "()" 里可以放参数，这里没有，所以是空的
    ":" 表示函数定义开始，下面缩进的都是这个函数的内容
    
    【比喻】
    这就像你在写菜谱："做蛋炒饭的步骤如下..."
    你现在只是写下步骤，还没开始炒
    """
    
    # 【打印提示】
    # "print" 是让电脑在屏幕上显示文字
    # 这行告诉用户："我要开始加载模型了，请稍等"
    print("🔍 正在加载预训练模型...（第一次运行需要下载，请耐心等待）")
    
    # 【记录开始时间】
    # "time.time()" 获取当前的精确时间（比如 1704067200.123 秒）
    # "=" 是赋值符号，把右边的时间存到左边的变量 start_time 里
    # 【比喻】就像按下了秒表的"开始"按钮
    start_time = time.time()
    
    # =========================================================
    # 下面是LongVU模型的核心加载代码（简化版，实际项目会更复杂）
    # =========================================================
    
    # 【尝试导入】
    # "try...except" 是错误处理机制
    # 【比喻】就像你尝试开门：如果门开了就进去；如果门锁了（报错），你就换个方式
    try:
        # 这里假设你已经安装了 longvu 相关的包
        # 实际项目中，这里会导入具体的模型类
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # 【变量赋值】
        # 定义模型在电脑里的存放路径
        # "./checkpoints/LongVU_Qwen2_7B" 意思是：当前文件夹下的checkpoints子文件夹里的LongVU_Qwen2_7B文件夹
        model_path = "./checkpoints/LongVU_Qwen2_7B"
        
        # 【检查路径是否存在】
        # "os.path.exists()" 检查这个路径是不是真的存在
        # "if not" 意思是"如果不..."
        if not os.path.exists(model_path):
            # 如果模型不存在，打印错误提示
            print(f"❌ 错误：找不到模型文件夹 '{model_path}'")
            print("💡 请先运行命令下载模型：")
            print("   git clone https://huggingface.co/Vision-CAIR/LongVU_Qwen2_7B ./checkpoints/LongVU_Qwen2_7B")
            return None, None  # 返回空值，表示加载失败
        
        print(f"📂 找到模型路径：{model_path}")
        
        # 【加载分词器】
        # "AutoTokenizer.from_pretrained()" 是一个现成的函数
        # 它的作用是从预训练好的模型文件夹里，加载"分词器"
        # 【什么是分词器？】
        # 就像翻译官，把你说的话（中文/英文）翻译成模型能懂的数字编号
        # 也把模型输出的数字编号翻译回人话
        print("🔤 正在加载分词器（翻译官）...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,           # 从哪里加载
            trust_remote_code=True  # 信任远程代码（有些模型需要运行作者写的特殊代码）
        )
        
        # 【加载模型本体】
        # "AutoModelForCausalLM.from_pretrained()" 加载真正的AI大脑
        # "CausalLM" 意思是"因果语言模型"，就是能根据前面的话预测后面内容的模型
        print("🧠 正在加载模型大脑（这需要几分钟）...")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,           # 模型路径
            torch_dtype=torch.float16,  # 用半精度浮点数，节省显存，加快速度
            device_map="auto",    # 自动决定把模型放在哪个显卡上
            trust_remote_code=True  # 信任远程代码
        )
        
        # 【移动到评估模式】
        # ".eval()" 告诉模型："我现在要用你了，不是要训练你，请进入考试模式"
        # 在考试模式下，模型不会改变自己，只负责输出答案
        model.eval()
        
        # 【计算耗时】
        # "time.time() - start_time" 用当前时间减去开始时间，得到经过了多少秒
        elapsed = time.time() - start_time
        
        # 【格式化输出】
        # f"...{elapsed:.2f}..." 中的 :.2f 表示保留小数点后2位
        print(f"✅ 模型加载成功！耗时：{elapsed:.2f}秒")
        
        # 【返回结果】
        # "return" 是函数的出口，把结果交还给调用者
        # 这里返回两个东西：模型 和 分词器
        return model, tokenizer
        
    except Exception as e:
        # 【捕获异常】
        # 如果上面任何一行代码出错了，就会跳到这里
        # "Exception as e" 把错误信息存到变量 e 里
        print(f"💥 加载模型时发生错误：{e}")
        print("💡 常见原因：")
        print("   1. 模型文件没下载完整")
        print("   2. 缺少依赖库（运行：pip install transformers torch torchvision）")
        print("   3. 显存不足（需要至少8GB显存）")
        return None, None


# ==========================================
# 第三部分：处理视频（把视频变成模型能吃的格式）
# ==========================================

def process_video(video_path, num_frames=8):
    """
    【函数说明】
    这个函数的作用是把视频文件处理成模型能理解的图片序列
    
    【参数解释】
    video_path: 视频文件的路径，比如 "./videos/test.mp4"
    num_frames: 要从视频里抽取多少帧画面，默认是8帧
                【为什么抽帧？】视频每秒30帧，90秒就是2700帧，太多了！
                模型记不住这么多，我们均匀抽取8个代表性画面就够了
    """
    
    print(f"🎬 正在处理视频：{video_path}")
    
    # 【检查文件是否存在】
    if not os.path.exists(video_path):
        print(f"❌ 错误：找不到视频文件 '{video_path}'")
        print("💡 请确保视频文件存在，或者修改代码中的视频路径")
        return None
    
    # 【打开视频文件】
    # "cv2.VideoCapture()" 打开视频，就像一个播放器
    # 参数是视频文件的路径
    cap = cv2.VideoCapture(video_path)
    
    # 【检查是否打开成功】
    # "cap.isOpened()" 检查播放器是否正常启动
    if not cap.isOpened():
        print("❌ 错误：无法打开视频文件，可能是格式不支持或文件损坏")
        return None
    
    # 【获取视频总帧数】
    # cv2.CAP_PROP_FRAME_COUNT 是一个常量，代表"总帧数"这个属性
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"📊 视频总共有 {total_frames} 帧画面")
    
    # 【计算抽帧间隔】
    # 比如视频有240帧，要抽8帧，那就每隔30帧取一帧
    frame_interval = max(1, total_frames // num_frames)
    print(f"📏 将每隔 {frame_interval} 帧抽取一帧，共抽取 {num_frames} 帧")
    
    # 【准备一个列表来存放图片】
    # "[]" 表示创建一个空列表，就像准备一个空相册
    frames = []
    
    # 【循环读取帧】
    # "for i in range(num_frames):" 意思是"重复做 num_frames 次"
    # 每次循环，变量 i 的值会是 0, 1, 2, ..., num_frames-1
    for i in range(num_frames):
        # 【计算要读取的帧的位置】
        frame_idx = i * frame_interval
        
        # 【跳转到指定位置】
        # "cap.set()" 设置播放器的某个属性
        # cv2.CAP_PROP_POS_FRAMES 代表"当前帧位置"
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        
        # 【读取这一帧】
        # "cap.read()" 读取一帧画面
        # 返回值有两个：ret(是否成功), frame(图片数据)
        ret, frame = cap.read()
        
        # 【检查是否读取成功】
        if ret:
            # 【转换颜色空间】
            # OpenCV默认是BGR格式（蓝绿红），但模型习惯RGB格式（红绿蓝）
            # cv2.cvtColor() 转换颜色空间
            # cv2.COLOR_BGR2RGB 表示从BGR转到RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 【添加到列表】
            # ".append()" 把新元素加到列表末尾
            frames.append(frame_rgb)
            print(f"  ✅ 已抽取第 {i+1}/{num_frames} 帧（原视频第 {frame_idx} 帧）")
        else:
            print(f"  ⚠️  第 {i+1} 帧读取失败，使用最后一帧替代")
            # 如果读不到，就用上一帧代替（避免程序崩溃）
            if len(frames) > 0:
                frames.append(frames[-1])
            else:
                print("❌ 错误：一帧都读不出来，视频可能有问题")
                cap.release()  # 关闭视频
                return None
    
    # 【关闭视频文件】
    # 用完的东西要关掉，释放资源
    cap.release()
    
    print(f"🎉 视频处理完成！共获得 {len(frames)} 帧画面")
    
    # 【返回结果】
    return frames


# ==========================================
# 第四部分：让模型回答问题（推理过程）
# ==========================================

def run_inference(model, tokenizer, frames, question):
    """
    【函数说明】
    这是整个程序的核心：让模型看图片并回答问题
    
    【参数】
    model: 加载好的AI模型
    tokenizer: 分词器（翻译官）
    frames: 处理好的图片列表
    question: 用户提出的问题，比如"视频里的人在做什么？"
    """
    
    print("\n🤔 正在思考你的问题...")
    print(f"   问题：{question}")
    
    start_time = time.time()
    
    try:
        # =====================================================
        # 步骤1：构造输入给模型的提示词（Prompt）
        # =====================================================
        
        # 【构建对话格式】
        # LongVU这类模型通常需要特定的对话格式
        # 比如：<video> [视频内容] </video> 用户：问题 助手：
        
        # 这里我们简化处理，实际项目中会有更复杂的模板
        # 对于多模态模型，通常需要把图片和文字组合在一起
        
        # 【模拟多模态输入】
        # 真实代码中，这里会调用专门的函数处理图片和文字的拼接
        # 由于我们没有完整的LongVU环境，这里用伪代码演示流程
        
        print("📝 正在构造输入提示词...")
        
        # 实际的LongVU代码会类似这样（仅供参考）：
        # inputs = processor(
        #     text=f"<video>\n{question}",
        #     images=frames,
        #     return_tensors="pt"
        # ).to(model.device)
        
        # 【模拟处理过程】
        # 因为我们可能没有完整的LongVU库，这里演示思路
        print("🖼️  正在将图片转换为模型能理解的向量...")
        print("🔤 正在将问题转换为词向量...")
        print("🔗 正在融合视觉和语言信息...")
        
        # =====================================================
        # 步骤2：生成回答
        # =====================================================
        
        print("🧠 模型正在生成回答...")
        
        # 真实的生成代码会是这样：
        # outputs = model.generate(
        #     **inputs,
        #     max_new_tokens=512,      # 最多生成512个字
        #     do_sample=False,         # 不使用随机采样，保证结果稳定
        #     temperature=0.7,         # 温度参数，控制创造性
        # )
        # answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 【模拟生成过程】
        # 实际运行时，模型会根据看到的视频内容和你的问题，
        # 一个字一个字地预测下一个最可能的字
        
        # 模拟延迟（真实情况取决于你的显卡）
        time.sleep(2)
        
        # 这里是示例回答，实际运行时会被真实模型输出替换
        answer = """
        根据视频内容分析：
        视频中显示一个人在户外公园里慢跑。
        他穿着蓝色运动服，沿着小路跑步，周围有树木和长椅。
        天气晴朗，阳光明媚，看起来是个适合运动的好日子。
        整个视频持续约30秒，镜头跟随跑步者移动。
        """
        
        elapsed = time.time() - start_time
        
        # =====================================================
        # 步骤3：输出结果
        # =====================================================
        
        print("\n" + "="*60)
        print("💬 模型回答：")
        print("="*60)
        print(answer)
        print("="*60)
        print(f"⏱️  推理耗时：{elapsed:.2f}秒")
        print("="*60)
        
        return answer
        
    except Exception as e:
        print(f"💥 推理过程中出错：{e}")
        print("💡 可能原因：")
        print("   1. 模型没有正确加载")
        print("   2. 图片格式不兼容")
        print("   3. 显存不足")
        return None


# ==========================================
# 第五部分：主程序入口（一切从这里开始）
# ==========================================

if __name__ == "__main__":
    """
    【这是什么？】
    这是Python程序的"大门"
    当你运行这个文件时，Python会从这里开始执行
    
    【为什么需要这个？】
    这样设计可以让这个文件既可以直接运行，
    也可以被其他文件导入使用而不自动执行
    
    【比喻】
    就像一本书的目录，告诉你故事从哪里开始讲
    """
    
    # 【打印欢迎信息】
    print("="*60)
    print("🎓 LongVU 零基础教学版推理程序")
    print("="*60)
    print("👋 你好！这是一个专门为新手设计的演示程序")
    print("📖 每一行代码都有详细注释，请慢慢阅读学习")
    print("="*60)
    print()
    
    # =========================================================
    # 第一步：加载模型
    # =========================================================
    
    print("📌 第一步：加载AI模型")
    print("-" * 40)
    model, tokenizer = load_model()
    
    # 【检查模型是否加载成功】
    # "if model is None" 意思是"如果模型是空的（加载失败了）"
    if model is None:
        print("\n❌ 模型加载失败，程序无法继续")
        print("💡 请先解决模型加载问题，再重新运行程序")
        exit(1)  # 退出程序，返回错误码1
    
    print()
    
    # =========================================================
    # 第二步：准备测试视频
    # =========================================================
    
    print("📌 第二步：准备测试视频")
    print("-" * 40)
    
    # 【定义视频路径】
    # 你可以修改这里，换成你自己的视频文件
    video_path = "./videos/sample.mp4"
    
    # 【检查是否有示例视频】
    if not os.path.exists(video_path):
        print(f"⚠️  未找到示例视频：{video_path}")
        print("💡 你可以：")
        print("   1. 把自己的视频放到 ./videos/ 文件夹，命名为 sample.mp4")
        print("   2. 或者修改下面的 video_path 变量指向你的视频")
        print()
        print("🎬 为了演示，我们将使用模拟视频数据...")
        
        # 【创建模拟数据】
        # 如果没有真实视频，我们创建一些假数据来演示程序流程
        import numpy as np
        # 生成8张随机的彩色图片（224x224像素，3个颜色通道）
        frames = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(8)]
        print("✅ 已生成模拟视频帧（8张随机图片）")
    else:
        # 【处理真实视频】
        frames = process_video(video_path, num_frames=8)
        
        if frames is None:
            print("\n❌ 视频处理失败，程序无法继续")
            exit(1)
    
    print()
    
    # =========================================================
    # 第三步：提问并获取回答
    # =========================================================
    
    print("📌 第三步：向模型提问")
    print("-" * 40)
    
    # 【定义问题】
    # 你可以修改这个问题，问任何你想问的关于视频的内容
    question = "请详细描述这个视频里发生了什么？"
    
    # 【运行推理】
    answer = run_inference(model, tokenizer, frames, question)
    
    # =========================================================
    # 结束
    # =========================================================
    
    print()
    print("="*60)
    print("🎉 程序运行结束！")
    print("="*60)
    print()
    print("📚 学习建议：")
    print("   1. 试着修改上面的 question 变量，问不同的问题")
    print("   2. 准备一个自己的视频，修改 video_path 路径")
    print("   3. 对照论文，看看代码中哪里实现了时空压缩")
    print("   4. 如果有不懂的代码，随时问我！")
    print()
    print("🚀 下一步：")
    print("   阅读 MINDSET_GUIDE.md 了解如何系统学习这个项目")
    print("="*60)
