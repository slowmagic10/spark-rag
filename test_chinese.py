#!/usr/bin/env python3
"""
测试中文强制回答功能
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from gradio_chat_app import RAGChatBot

def test_chinese_responses():
    """测试强制中文回答功能"""
    print("🧪 测试强制中文回答功能...")
    
    # 初始化 RAG 聊天机器人
    rag_bot = RAGChatBot()
    
    # 检查服务器健康状态
    if not rag_bot.check_health():
        print("❌ RAG 服务器未响应，无法进行测试")
        return
    
    print("✅ RAG 服务器连接正常")
    
    # 测试英文问题，期望中文回答
    test_cases = [
        {
            "question": "What is artificial intelligence?",
            "description": "英文问题测试"
        },
        {
            "question": "How does machine learning work?",
            "description": "另一个英文问题测试"
        },
        {
            "question": "你好，请介绍一下深度学习",
            "description": "中文问题测试"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📤 测试 {i}: {test_case['description']}")
        print(f"❓ 问题: {test_case['question']}")
        
        # 准备带系统指令的消息
        test_messages = rag_bot.format_messages_for_api([], test_case['question'])
        
        print("💬 AI 回复 (应为中文):")
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
            
            # 检查回答是否包含中文
            full_response = ''.join(response_parts)
            has_chinese = any('\u4e00' <= char <= '\u9fff' for char in full_response)
            
            if has_chinese:
                print(f"✅ 测试 {i} 通过: 回答包含中文")
            else:
                print(f"⚠️  测试 {i} 警告: 回答可能不是中文")
                
        except Exception as e:
            print(f"\n❌ 测试 {i} 失败: {str(e)}")
    
    print(f"\n🏁 中文回答测试完成")

if __name__ == "__main__":
    test_chinese_responses()
