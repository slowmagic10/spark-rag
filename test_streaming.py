#!/usr/bin/env python3
"""
测试流式输出功能
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from gradio_chat_app import RAGChatBot

def test_streaming():
    """测试流式输出功能"""
    print("🧪 测试流式输出功能...")
    
    # 初始化 RAG 聊天机器人
    rag_bot = RAGChatBot()
    
    # 检查服务器健康状态
    if not rag_bot.check_health():
        print("❌ RAG 服务器未响应，无法进行流式测试")
        return
    
    print("✅ RAG 服务器连接正常")
    
    # 准备测试消息
    test_messages = [
        {"role": "user", "content": "你好，请介绍一下人工智能"}
    ]
    
    print("📤 发送流式请求...")
    print("💬 AI 回复 (流式):")
    print("-" * 50)
    
    try:
        # 测试流式输出
        response_parts = []
        for chunk in rag_bot.query_rag_stream(
            messages=test_messages,
            temperature=0.1,
            use_knowledge_base=False
        ):
            print(chunk, end='', flush=True)
            response_parts.append(chunk)
        
        print("\n" + "-" * 50)
        print(f"✅ 流式输出测试完成，共接收 {len(response_parts)} 个数据块")
        
        if response_parts:
            print(f"📄 完整回复: {''.join(response_parts)}")
        
    except Exception as e:
        print(f"\n❌ 流式输出测试失败: {str(e)}")

if __name__ == "__main__":
    test_streaming()
