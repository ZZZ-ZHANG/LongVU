#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LongVU 快速推理测试脚本
适合新手入门，测试模型是否能正常运行
"""

import numpy as np
import torch
from longvu.builder import load_pretrained_model
from longvu.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from longvu.conversation import conv_templates, SeparatorStyle
from longvu.mm_datautils import (
    KeywordsStoppingCriteria,
    process_images,
    tokenizer_image_token,
)
from decord import cpu, VideoReader

def main():
    print("=" * 60)
    print("LongVU 推理测试 - 新手友好版")
    print("=" * 60)
    
    # 1. 加载预训练模型
    print("\n[1/5] 正在加载模型...（首次加载较慢，请耐心等待）")
    model_path = "./checkpoints/longvu_qwen"  # 确保这里是你下载的路径
    
    try:
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path, None, "cambrian_qwen"
        )
        print("✓ 模型加载成功！")
    except Exception as e:
        print(f"✗ 模型加载失败：{e}")
        print("请检查：1) checkpoints 目录下是否有模型文件 2) 路径是否正确")
        return
    
    model.eval()
    print(f"   模型设备：{model.device}")
    
    # 2. 准备测试视频
    print("\n[2/5] 正在加载测试视频...")
    video_path = "./examples/video1.mp4"
    
    try:
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        print(f"✓ 视频加载成功！总帧数：{len(vr)}")
    except Exception as e:
        print(f"✗ 视频加载失败：{e}")
        return
    
    # 3. 抽取帧（每秒 1 帧）
    print("\n[3/5] 正在抽取视频帧...")
    fps = float(vr.get_avg_fps())
    print(f"   视频 FPS: {fps:.2f}")
    
    frame_indices = np.array([i for i in range(0, len(vr), max(1, round(fps)))])
    print(f"   抽取帧数：{len(frame_indices)}")
    
    video_frames = []
    for frame_index in frame_indices:
        img = vr[frame_index].asnumpy()
        video_frames.append(img)
    video_frames = np.stack(video_frames)
    
    # 4. 预处理图像
    print("\n[4/5] 正在预处理图像...")
    image_sizes = [video_frames[0].shape[:2]]
    video_processed = process_images(video_frames, image_processor, model.config)
    video_processed = [item.unsqueeze(0) for item in video_processed]
    print("✓ 预处理完成！")
    
    # 5. 构建 prompt 并生成回答
    print("\n[5/5] 正在生成描述...")
    qs = "Describe this video in detail"
    qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
    
    conv = conv_templates["qwen"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(model.device)
    
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=video_processed,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0.2,
            max_new_tokens=256,  # 增加输出长度
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )
    
    pred = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    
    # 6. 显示结果
    print("\n" + "=" * 60)
    print("🎉 推理成功！模型输出：")
    print("=" * 60)
    print(pred)
    print("=" * 60)

if __name__ == "__main__":
    main()
