#!/usr/bin/env python3
"""
NVIDIA RAG Gradio Chat App with Conversation History
"""

import gradio as gr
import requests
import json
import os
from typing import List, Tuple, Dict, Any
import time

class RAGChatBot:
    def __init__(self, base_url: str = "http://192.168.81.253:8081/v1"):
        """
        Initialize the RAG ChatBot
        
        Args:
            base_url: The base URL of the RAG server (default: http://192.168.81.253:8081/v1)
        """
        self.base_url = base_url
        self.generate_url = f"{base_url}/generate"
        self.health_url = f"{base_url}/health"
        
    def check_health(self) -> bool:
        """Check if the RAG server is healthy"""
        try:
            response = requests.get(self.health_url, timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def format_messages_for_api(self, history: List[List[str]], new_message: str) -> List[Dict[str, str]]:
        """
        Convert Gradio chat history to API format with Chinese language instruction
        
        Args:
            history: Gradio chat history format [(user_msg, bot_msg), ...]
            new_message: The new user message
            
        Returns:
            List of messages in API format with system instruction
        """
        messages = []
        
        # 添加系统指令确保使用中文回答
        system_instruction = {
            "role": "system", 
            "content": "你是一个智能助手，请始终使用中文回答所有问题。无论用户使用什么语言提问，你都必须用中文进行回答。请确保回答准确、详细且有帮助。"
        }
        messages.append(system_instruction)
        
        # Add conversation history
        for user_msg, bot_msg in history:
            if user_msg:
                messages.append({"role": "user", "content": user_msg})
            if bot_msg:
                messages.append({"role": "assistant", "content": bot_msg})
        
        # Add the new user message
        messages.append({"role": "user", "content": new_message})
        
        return messages
    
    def query_rag_stream(self, messages: List[Dict[str, str]], 
                         temperature: float = 0.1, 
                         top_p: float = 0.9, 
                         max_tokens: int = 4096,
                         use_knowledge_base: bool = True):
        """
        Query the RAG API with streaming response
        
        Args:
            messages: List of messages in API format
            temperature: Sampling temperature (0-1)
            top_p: Top-p sampling parameter (0.1-1)
            max_tokens: Maximum tokens to generate
            use_knowledge_base: Whether to use knowledge base
            
        Yields:
            Streaming response chunks
        """
        # 使用与前端相同的配置格式
        payload = {
            "messages": messages,
            "collection_names": ["test"] if use_knowledge_base else [],
            "temperature": temperature,
            "top_p": top_p,
            "reranker_top_k": 10,
            "vdb_top_k": 10,
            "confidence_threshold": 0.5,
            "use_knowledge_base": use_knowledge_base,
            "enable_citations": True,
            "enable_guardrails": False
        }
        
        try:
            response = requests.post(
                self.generate_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                stream=True,  # 启用流式响应
                timeout=60
            )
            
            if response.status_code == 200:
                # 处理流式响应
                for line in response.iter_lines():
                    if line:
                        line_text = line.decode('utf-8')
                        if line_text.startswith('data: '):
                            try:
                                data = json.loads(line_text[6:])
                                if 'choices' in data and data['choices']:
                                    delta = data['choices'][0].get('delta', {})
                                    if 'content' in delta:
                                        yield delta['content']
                            except json.JSONDecodeError:
                                continue
                        elif line_text.strip() == 'data: [DONE]':
                            break
            else:
                yield f"❌ 服务器返回状态码 {response.status_code}: {response.text[:200]}"
                
        except requests.exceptions.ConnectionError:
            yield "❌ 无法连接到RAG服务器。请确保服务器正在运行。"
        except requests.exceptions.Timeout:
            yield "❌ 请求超时，请稍后重试。"
        except Exception as e:
            yield f"❌ 未知错误: {str(e)}"
    
    def query_rag_fallback(self, messages: List[Dict[str, str]], 
                          temperature: float = 0.1, 
                          top_p: float = 0.9, 
                          max_tokens: int = 4096,
                          use_knowledge_base: bool = True) -> str:
        """
        Fallback non-streaming query method
        """
        configs = [
            {
                "messages": messages,
                "collection_names": ["test"] if use_knowledge_base else [],
                "temperature": temperature,
                "top_p": top_p,
                "reranker_top_k": 10,
                "vdb_top_k": 10,
                "confidence_threshold": 0.5,
                "use_knowledge_base": use_knowledge_base,
                "enable_citations": True,
                "enable_guardrails": False
            },
            {
                "messages": messages,
                "collection_names": [],
                "temperature": 0.1,
                "top_p": 0.9,
                "reranker_top_k": 10,
                "vdb_top_k": 10,
                "confidence_threshold": 0.5,
                "use_knowledge_base": False,
                "enable_citations": False,
                "enable_guardrails": False
            }
        ]
        
        for i, payload in enumerate(configs, 1):
            try:
                response = requests.post(
                    self.generate_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=60
                )
                
                if response.status_code == 200:
                    try:
                        result = response.json()
                        if "choices" in result and len(result["choices"]) > 0:
                            content = result["choices"][0]["message"]["content"]
                            return content
                    except json.JSONDecodeError:
                        response_text = response.text
                        if response_text.strip():
                            lines = response_text.strip().split('\n')
                            content_parts = []
                            for line in lines:
                                if line.startswith('data: '):
                                    try:
                                        data = json.loads(line[6:])
                                        if 'choices' in data and data['choices']:
                                            delta = data['choices'][0].get('delta', {})
                                            if 'content' in delta:
                                                content_parts.append(delta['content'])
                                    except:
                                        continue
                            if content_parts:
                                return ''.join(content_parts)
                        return "✅ 连接成功，但响应格式未知"
                elif response.status_code == 500:
                    if i < len(configs):
                        continue
                    else:
                        return f"❌ 服务器内部错误 (500)"
                else:
                    return f"❌ 服务器返回状态码 {response.status_code}"
            except Exception as e:
                if i < len(configs):
                    continue
                else:
                    return f"❌ 未知错误: {str(e)}"
        
        return "❌ 所有配置都失败了，请检查服务器状态。"
        """
        Query the RAG API with conversation history
        
        Args:
            messages: List of messages in API format
            temperature: Sampling temperature (0-1)
            top_p: Top-p sampling parameter (0.1-1)
            max_tokens: Maximum tokens to generate
            use_knowledge_base: Whether to use knowledge base
            
        Returns:
            The generated response
        """
        # 使用与前端相同的配置格式
        configs = [
            # 配置1: 完整参数 (类似前端)
            {
                "messages": messages,
                "collection_names": ["test"] if use_knowledge_base else [],
                "temperature": temperature,
                "top_p": top_p,
                "reranker_top_k": 10,
                "vdb_top_k": 10,
                "confidence_threshold": 0.5,
                "use_knowledge_base": use_knowledge_base,
                "enable_citations": True,
                "enable_guardrails": False
            },
            # 配置2: 简化参数
            {
                "messages": messages,
                "collection_names": [],
                "temperature": 0.1,
                "top_p": 0.9,
                "reranker_top_k": 10,
                "vdb_top_k": 10,
                "confidence_threshold": 0.5,
                "use_knowledge_base": False,
                "enable_citations": False,
                "enable_guardrails": False
            },
            # 配置3: 最简配置
            {
                "messages": messages,
                "use_knowledge_base": False,
                "temperature": 0.1,
                "max_tokens": min(max_tokens, 1000)
            }
        ]
        
        for i, payload in enumerate(configs, 1):
            try:
                response = requests.post(
                    self.generate_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=60
                )
                
                if response.status_code == 200:
                    try:
                        # 尝试解析 JSON 响应
                        result = response.json()
                        if "choices" in result and len(result["choices"]) > 0:
                            content = result["choices"][0]["message"]["content"]
                            if i > 1:
                                content = f"⚠️ 使用简化模式回答 (配置{i}):\n\n{content}"
                            return content
                    except json.JSONDecodeError:
                        # 可能是流式响应，尝试直接读取文本
                        response_text = response.text
                        if response_text.strip():
                            # 简单处理流式响应
                            lines = response_text.strip().split('\n')
                            content_parts = []
                            for line in lines:
                                if line.startswith('data: '):
                                    try:
                                        data = json.loads(line[6:])
                                        if 'choices' in data and data['choices']:
                                            delta = data['choices'][0].get('delta', {})
                                            if 'content' in delta:
                                                content_parts.append(delta['content'])
                                    except:
                                        continue
                            if content_parts:
                                content = ''.join(content_parts)
                                if i > 1:
                                    content = f"⚠️ 使用简化模式回答 (配置{i}):\n\n{content}"
                                return content
                        return f"✅ 连接成功，但响应格式未知 (配置{i})"
                        
                elif response.status_code == 500:
                    if i < len(configs):
                        continue  # 尝试下一个配置
                    else:
                        return f"❌ 服务器内部错误 (500)。可能的原因:\n" \
                               f"• 模型服务未完全启动\n" \
                               f"• 缺少必要的依赖服务\n" \
                               f"• 配置问题\n\n" \
                               f"详细错误: {response.text[:200]}"
                else:
                    return f"❌ 服务器返回状态码 {response.status_code}\n{response.text[:200]}"
                    
            except requests.exceptions.Timeout:
                if i < len(configs):
                    continue  # 尝试下一个配置
                else:
                    return "❌ 请求超时，请稍后重试。"
            except requests.exceptions.ConnectionError:
                return "❌ 无法连接到RAG服务器。请确保服务器正在运行。"
            except Exception as e:
                if i < len(configs):
                    continue  # 尝试下一个配置
                else:
                    return f"❌ 未知错误: {str(e)}"
        
        return "❌ 所有配置都失败了，请检查服务器状态。"

def create_gradio_interface():
    """Create and configure the Gradio interface"""
    
    # Initialize the chatbot
    rag_bot = RAGChatBot()
    
    def chat_fn(message: str, history: List[List[str]], 
                temperature: float, top_p: float, max_tokens: int, 
                use_knowledge_base: bool, force_chinese: bool):
        """
        Handle chat interaction with streaming support
        
        Args:
            message: User input message
            history: Chat history
            temperature: Sampling temperature
            top_p: Top-p sampling
            max_tokens: Maximum tokens
            use_knowledge_base: Whether to use knowledge base
            force_chinese: Whether to force Chinese responses
            
        Yields:
            Updated history and empty string for message box
        """
        if not message.strip():
            yield history, ""
            return
        
        # 立即显示用户消息
        history = history + [[message, ""]]
        yield history, ""
        
        # Check server health
        if not rag_bot.check_health():
            error_msg = "❌ RAG服务器未响应。请确保服务器正在运行在 http://192.168.81.253:8081"
            history[-1][1] = error_msg
            yield history, ""
            return
        
        # Convert to API format (系统指令会在 format_messages_for_api 中自动添加)
        api_messages = rag_bot.format_messages_for_api(history[:-1], message)
        
        # 如果不强制中文，移除系统指令
        if not force_chinese and api_messages and api_messages[0]["role"] == "system":
            api_messages = api_messages[1:]
        
        # Get streaming response from RAG
        try:
            assistant_message = ""
            for chunk in rag_bot.query_rag_stream(
                messages=api_messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                use_knowledge_base=use_knowledge_base
            ):
                assistant_message += chunk
                # 更新最后一条消息的回复部分
                history[-1][1] = assistant_message
                yield history, ""
                time.sleep(0.02)  # 小延迟以实现打字效果
                
        except Exception as e:
            error_msg = f"❌ 处理请求时发生错误: {str(e)}"
            history[-1][1] = error_msg
            yield history, ""
    
    def clear_history():
        """Clear chat history"""
        return [], ""
    
    def check_server_status():
        """Check and return server status"""
        if rag_bot.check_health():
            return "✅ RAG服务器状态: 正常"
        else:
            return "❌ RAG服务器状态: 无响应"
    
    # Create the Gradio interface
    with gr.Blocks(
        title="NVIDIA RAG 聊天机器人",
        theme=gr.themes.Soft(),
        css="""
        /* 全局字体设置 */
        * {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        }
        
        /* 聊天容器 */
        .chat-container {
            max-height: 600px;
            overflow-y: auto;
        }
        
        /* 流式输出动画 */
        @keyframes typing {
            0% { opacity: 0.7; }
            50% { opacity: 1; }
            100% { opacity: 0.7; }
        }
        
        /* 正在输入指示器 */
        .typing-indicator::after {
            content: '●';
            animation: typing 1.5s infinite;
            color: #22c55e;
            margin-left: 4px;
        }
        
        /* 标题字体 */
        h1, h2, h3, h4, h5, h6 {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            font-weight: 600;
        }
        
        /* 输入框字体 */
        .gr-textbox input, .gr-textbox textarea {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            font-size: 14px;
        }
        
        /* 按钮字体 */
        .gr-button {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            font-weight: 500;
        }
        
        /* 聊天气泡字体 */
        .message {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            font-size: 14px;
            line-height: 1.5;
        }
        
        /* 标签字体 */
        .gr-form label {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            font-weight: 500;
            font-size: 13px;
        }
        
        /* 滑块标签 */
        .gr-slider label {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            font-size: 13px;
        }
        
        /* Markdown 内容 */
        .markdown {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            line-height: 1.6;
        }
        
        /* 代码字体 */
        code, pre {
            font-family: "SF Mono", Monaco, "Cascadia Code", "Roboto Mono", Consolas, "Courier New", monospace;
        }
        
        /* 聊天气泡优化 */
        .chatbot .message-wrap {
            margin-bottom: 12px;
        }
        
        .chatbot .message {
            border-radius: 12px;
            padding: 12px 16px;
            margin: 4px 0;
        }
        """
    ) as demo:
        
        gr.Markdown(
            """
            # NVIDIA RAG 智能问答系统
            
            基于 NVIDIA RAG Blueprint 构建的企业级智能问答系统，支持多轮对话和知识库检索。
            
            **核心功能:**
            • 多轮对话历史记忆
            • 基于向量数据库的知识检索
            • 可调节的生成参数
            • 实时服务状态监控
            """ + (f"\n\n🔥 **开发模式**: 热重载已启用" if os.environ.get('GRADIO_RELOAD') == 'true' else ""),
            elem_classes=["markdown"]
        )
        
        with gr.Row():
            with gr.Column(scale=2):
                # Chat interface
                chatbot = gr.Chatbot(
                    label="对话历史",
                    height=500,
                    elem_classes=["chat-container"],
                    show_copy_button=True,
                    bubble_full_width=False,
                    sanitize_html=False,  # 允许 HTML 以支持更好的格式化
                    render_markdown=True   # 支持 Markdown 渲染
                )
                
                with gr.Row():
                    msg_box = gr.Textbox(
                        label="输入您的问题",
                        placeholder="请输入您想要询问的问题...",
                        scale=4,
                        show_label=False
                    )
                    send_btn = gr.Button("发送", variant="primary", scale=1)
                
                with gr.Row():
                    clear_btn = gr.Button("清空对话", variant="secondary")
                    status_btn = gr.Button("检查服务器状态", variant="secondary")
                
                server_status = gr.Textbox(
                    label="服务器状态",
                    value="点击'检查服务器状态'来检查连接",
                    interactive=False
                )
            
            with gr.Column(scale=1):
                # Parameters panel
                gr.Markdown("### 生成参数", elem_classes=["markdown"])
                
                use_kb = gr.Checkbox(
                    label="启用知识库",
                    value=True,
                    info="基于向量数据库内容生成回答"
                )
                
                force_chinese = gr.Checkbox(
                    label="强制中文回答",
                    value=True,
                    info="确保所有回答都使用中文，无论问题使用什么语言"
                )
                
                temperature = gr.Slider(
                    label="随机性 (Temperature)",
                    minimum=0.0,
                    maximum=1.0,
                    value=0.1,
                    step=0.1,
                    info="控制回答的随机性，值越小回答越确定"
                )
                
                top_p = gr.Slider(
                    label="核采样 (Top-p)",
                    minimum=0.1,
                    maximum=1.0,
                    value=0.9,
                    step=0.1,
                    info="控制词汇选择的多样性"
                )
                
                max_tokens = gr.Slider(
                    label="最大长度",
                    minimum=256,
                    maximum=8192,
                    value=4096,
                    step=256,
                    info="生成回答的最大字符数"
                )
                
                gr.Markdown(
                    """
                    ### 使用说明
                    
                    **连接要求**  
                    确保 RAG 服务运行在 `192.168.81.253:8081`
                    
                    **操作步骤**  
                    1. 在下方输入框中输入问题（支持中英文）
                    2. 点击发送或按回车键提交
                    3. 系统将基于知识库生成回答
                    4. 支持多轮对话上下文记忆
                    
                    **语言设置**  
                    • **强制中文回答**: 启用后所有回答都使用中文
                    • 无论用户使用中文或英文提问，AI 都会用中文回答
                    • 关闭此选项后，AI 会根据问题语言自动选择回答语言
                    
                    **参数调节**  
                    • **随机性**: 控制回答的创造性
                    • **核采样**: 影响用词选择
                    • **最大长度**: 限制回答长度
                    
                    **故障排除**  
                    如遇连接问题请检查服务器状态
                    """,
                    elem_classes=["markdown"]
                )
        
        # Event handlers with streaming support
        send_btn.click(
            fn=chat_fn,
            inputs=[msg_box, chatbot, temperature, top_p, max_tokens, use_kb, force_chinese],
            outputs=[chatbot, msg_box],
            show_progress=False  # 隐藏进度条以获得更好的流式体验
        )
        
        msg_box.submit(
            fn=chat_fn,
            inputs=[msg_box, chatbot, temperature, top_p, max_tokens, use_kb, force_chinese],
            outputs=[chatbot, msg_box],
            show_progress=False  # 隐藏进度条以获得更好的流式体验
        )
        
        clear_btn.click(
            fn=clear_history,
            outputs=[chatbot, msg_box]
        )
        
        status_btn.click(
            fn=check_server_status,
            outputs=[server_status]
        )
        
        # Initialize server status on load
        demo.load(
            fn=check_server_status,
            outputs=[server_status]
        )
    
    return demo

def main():
    """Main function to launch the Gradio app"""
    
    # 检查是否在重载模式
    is_reload_mode = os.environ.get('GRADIO_RELOAD', 'false').lower() == 'true'
    
    # Standalone execution
    print("启动 NVIDIA RAG 智能问答系统...")
    print("服务器地址: http://192.168.81.253:8081")
    
    if is_reload_mode:
        print("🔥 重载模式已启用")
        print("正在启动 Web 界面（支持热重载）...")
    else:
        print("正在启动 Web 界面...")
        
    demo = create_gradio_interface()
    
    launch_kwargs = {
        "server_name": "0.0.0.0",  # 允许外部访问
        "server_port": 7860,       # Gradio默认端口
        "share": False,            # 不创建公共链接
        "inbrowser": not is_reload_mode,  # 重载模式下不自动打开浏览器
        "quiet": is_reload_mode,   # 重载模式下减少输出
        "show_error": True         # 显示错误信息
    }
    
    # 在重载模式下启用额外功能
    if is_reload_mode:
        launch_kwargs["debug"] = True
        
    demo.launch(**launch_kwargs)

if __name__ == "__main__":
    main()
