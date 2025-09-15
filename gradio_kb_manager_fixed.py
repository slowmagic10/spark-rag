#!/usr/bin/env python3
"""
增强版 NVIDIA RAG Gradio 聊天应用
包含完整的知识库管理功能：
- 聊天界面（支持流式输出和对话历史）
- 知识库创建和删除
- 文档上传和删除
- 知识库选择
- 美化的中文界面
"""

import gradio as gr
import requests
import json
import os
import time
from typing import List, Dict, Any, Optional, Tuple
import mimetypes
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 配置
RAG_SERVER_URL = "http://192.168.81.253:8081"
INGESTOR_SERVER_URL = "http://192.168.81.253:8082"

# 系统提示词，强制使用中文回复
SYSTEM_PROMPT = """
你是一个专业的AI助手。请严格遵循以下规则：
1. 必须使用中文回复所有问题
2. 回复要准确、有用、详细
3. 保持礼貌和专业的语气
4. 基于提供的文档内容进行回答
"""

class KnowledgeBaseManager:
    """知识库管理器"""
    
    def __init__(self, ingestor_url: str):
        self.ingestor_url = ingestor_url
    
    def list_collections(self) -> List[str]:
        """获取所有知识库列表"""
        try:
            response = requests.get(f"{self.ingestor_url}/collections")
            if response.status_code == 200:
                data = response.json()
                # 根据 API 响应结构解析集合名称
                if isinstance(data, dict) and 'collections' in data:
                    # 处理包含详细信息的格式
                    collections = data['collections']
                    if isinstance(collections, list) and len(collections) > 0:
                        if isinstance(collections[0], dict):
                            return [col.get('collection_name', str(col)) for col in collections]
                        else:
                            return collections
                    return []
                elif isinstance(data, list):
                    # 处理简单列表格式
                    if len(data) > 0 and isinstance(data[0], dict):
                        return [col.get('collection_name', str(col)) for col in data]
                    return data
                else:
                    logger.warning(f"Unexpected collections response format: {data}")
                    return []
            else:
                logger.error(f"Failed to list collections: {response.status_code} - {response.text}")
                return []
        except Exception as e:
            logger.error(f"Error listing collections: {e}")
            return []
    
    def create_collection(self, collection_name: str, embedding_dimension: int = 2048) -> Tuple[bool, str]:
        """创建新的知识库"""
        try:
            payload = {
                "collection_name": collection_name,
                "embedding_dimension": embedding_dimension,
                "metadata_schema": []
            }
            
            response = requests.post(
                f"{self.ingestor_url}/collection",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                data = response.json()
                message = data.get('message', f'知识库 {collection_name} 创建成功')
                return True, message
            else:
                error_msg = f"创建知识库失败: {response.status_code} - {response.text}"
                logger.error(error_msg)
                return False, error_msg
                
        except Exception as e:
            error_msg = f"创建知识库时发生错误: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def delete_collection(self, collection_name: str) -> Tuple[bool, str]:
        """删除知识库"""
        try:
            response = requests.delete(
                f"{self.ingestor_url}/collections",
                json=[collection_name],
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                return True, f"知识库 {collection_name} 删除成功"
            else:
                error_msg = f"删除知识库失败: {response.status_code} - {response.text}"
                logger.error(error_msg)
                return False, error_msg
                
        except Exception as e:
            error_msg = f"删除知识库时发生错误: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def list_documents(self, collection_name: str) -> List[str]:
        """获取指定知识库中的文档列表"""
        try:
            response = requests.get(
                f"{self.ingestor_url}/documents",
                params={"collection_name": collection_name}
            )
            
            if response.status_code == 200:
                data = response.json()
                # 根据 API 响应结构解析文档列表
                if isinstance(data, dict) and 'documents' in data:
                    documents = data['documents']
                    if isinstance(documents, list):
                        return [doc.get('document_name', doc.get('name', doc.get('id', str(doc)))) for doc in documents]
                    return []
                elif isinstance(data, list):
                    return [doc.get('document_name', doc.get('name', doc.get('id', str(doc)))) for doc in data]
                else:
                    logger.warning(f"Unexpected documents response format: {data}")
                    return []
            else:
                logger.error(f"Failed to list documents: {response.status_code} - {response.text}")
                return []
                
        except Exception as e:
            logger.error(f"Error listing documents: {e}")
            return []
    
    def upload_document(self, file_path: str, collection_name: str) -> Tuple[bool, str]:
        """上传文档到指定知识库"""
        try:
            # 准备文件
            with open(file_path, 'rb') as f:
                files = {
                    'documents': (os.path.basename(file_path), f, self._get_mime_type(file_path))
                }
                
                # 准备数据
                data = {
                    'data': json.dumps({
                        "collection_name": collection_name,
                        "blocking": False,
                        "split_options": {
                            "chunk_size": 512,
                            "chunk_overlap": 150
                        },
                        "custom_metadata": [],
                        "generate_summary": False
                    })
                }
                
                response = requests.post(
                    f"{self.ingestor_url}/documents",
                    files=files,
                    data=data
                )
            
            if response.status_code == 200:
                data = response.json()
                task_id = data.get('task_id', 'unknown')
                return True, f"文档上传成功，任务ID: {task_id}（处理中...）"
            else:
                error_msg = f"文档上传失败: {response.status_code} - {response.text}"
                logger.error(error_msg)
                return False, error_msg
                
        except Exception as e:
            error_msg = f"上传文档时发生错误: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def delete_documents(self, collection_name: str, document_names: List[str]) -> Tuple[bool, str]:
        """从指定知识库删除文档"""
        try:
            response = requests.delete(
                f"{self.ingestor_url}/documents",
                params={"collection_name": collection_name},
                json=document_names,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                return True, f"成功删除 {len(document_names)} 个文档"
            else:
                error_msg = f"删除文档失败: {response.status_code} - {response.text}"
                logger.error(error_msg)
                return False, error_msg
                
        except Exception as e:
            error_msg = f"删除文档时发生错误: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def _get_mime_type(self, file_path: str) -> str:
        """获取文件的 MIME 类型"""
        mime_type, _ = mimetypes.guess_type(file_path)
        return mime_type or 'application/octet-stream'


class RAGChatBot:
    """RAG 聊天机器人"""
    
    def __init__(self, rag_server_url: str):
        self.rag_server_url = rag_server_url
        self.conversation_history = []
    
    def format_messages_for_api(self, user_message: str, history: List[List[str]], collection_name: str) -> Dict[str, Any]:
        """格式化消息为 API 所需的格式"""
        # 构建消息历史
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        # 添加历史对话
        for user_msg, assistant_msg in history:
            if user_msg:
                messages.append({"role": "user", "content": user_msg})
            if assistant_msg:
                messages.append({"role": "assistant", "content": assistant_msg})
        
        # 添加当前用户消息
        messages.append({"role": "user", "content": user_message})
        
        # 构建 API 请求体
        payload = {
            "query": user_message,
            "knowledge_base_name": collection_name,
            "chat_history": messages[1:],  # 排除 system 消息
            "use_knowledge_base": True,
            "embedding_config": {
                "model": "nvidia/nv-embedqa-e5-v5"
            },
            "llm_config": {
                "model": "nvidia/llama-3.1-nemotron-70b-instruct",
                "temperature": 0.1,
                "max_tokens": 1024,
                "top_p": 1.0,
                "stream": True
            }
        }
        
        return payload
    
    def query_rag_stream(self, payload: Dict[str, Any]):
        """流式查询 RAG API"""
        try:
            response = requests.post(
                f"{self.rag_server_url}/generate",
                json=payload,
                headers={"Content-Type": "application/json"},
                stream=True
            )
            
            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        line = line.decode('utf-8')
                        if line.startswith('data: '):
                            data = line[6:]
                            if data.strip() == '[DONE]':
                                break
                            try:
                                json_data = json.loads(data)
                                if 'choices' in json_data and len(json_data['choices']) > 0:
                                    delta = json_data['choices'][0].get('delta', {})
                                    content = delta.get('content', '')
                                    if content:
                                        yield content
                            except json.JSONDecodeError:
                                continue
            else:
                yield f"❌ API 请求失败: {response.status_code} - {response.text}"
                
        except Exception as e:
            yield f"❌ 连接错误: {str(e)}"


# 初始化管理器
kb_manager = KnowledgeBaseManager(INGESTOR_SERVER_URL)
chatbot = RAGChatBot(RAG_SERVER_URL)

# 全局状态
current_collection = "multimodal_data"

def get_collections_list():
    """获取知识库列表"""
    collections = kb_manager.list_collections()
    return collections if collections else ["multimodal_data"]

def refresh_collections():
    """刷新知识库列表"""
    collections = get_collections_list()
    first_collection = collections[0] if collections else None
    update_obj = gr.update(choices=collections, value=first_collection)
    return update_obj, update_obj, update_obj

def create_new_collection(collection_name: str):
    """创建新知识库"""
    if not collection_name.strip():
        return "❌ 请输入知识库名称", refresh_collections()[0]
    
    success, message = kb_manager.create_collection(collection_name.strip())
    
    if success:
        collections = get_collections_list()
        return f"✅ {message}", gr.update(choices=collections, value=collection_name.strip())
    else:
        return f"❌ {message}", refresh_collections()[0]

def delete_selected_collection(collection_name: str):
    """删除选中的知识库"""
    if not collection_name:
        return "❌ 请选择要删除的知识库", refresh_collections()[0]
    
    success, message = kb_manager.delete_collection(collection_name)
    
    if success:
        collections = get_collections_list()
        new_value = collections[0] if collections else None
        return f"✅ {message}", gr.update(choices=collections, value=new_value)
    else:
        return f"❌ {message}", refresh_collections()[0]

def get_documents_list(collection_name: str):
    """获取指定知识库的文档列表"""
    if not collection_name:
        return []
    
    documents = kb_manager.list_documents(collection_name)
    return documents

def refresh_documents(collection_name: str):
    """刷新文档列表"""
    documents = get_documents_list(collection_name)
    return gr.update(choices=documents)

def upload_files(files, collection_name: str):
    """上传文件到知识库"""
    if not files:
        return "❌ 请选择要上传的文件", gr.update()
    
    if not collection_name:
        return "❌ 请选择目标知识库", gr.update()
    
    results = []
    for file in files:
        success, message = kb_manager.upload_document(file.name, collection_name)
        results.append(f"📄 {os.path.basename(file.name)}: {message}")
    
    # 上传后刷新文档列表（延迟一下让处理完成）
    time.sleep(1)
    updated_documents = get_documents_list(collection_name)
    return "\n".join(results), gr.update(choices=updated_documents)

def delete_selected_documents(collection_name: str, selected_documents: List[str]):
    """删除选中的文档"""
    if not collection_name:
        return "❌ 请选择知识库", gr.update()
    
    if not selected_documents:
        return "❌ 请选择要删除的文档", gr.update()
    
    success, message = kb_manager.delete_documents(collection_name, selected_documents)
    
    if success:
        # 删除成功后刷新文档列表
        updated_documents = get_documents_list(collection_name)
        return f"✅ {message}", gr.update(choices=updated_documents, value=[])
    else:
        return f"❌ {message}", gr.update()

def update_current_collection(collection_name: str):
    """更新当前选中的知识库"""
    global current_collection
    current_collection = collection_name
    return f"✅ 已切换到知识库: {collection_name}"

def chat_fn(message: str, history: List[Dict[str, str]], collection_name: str):
    """聊天函数"""
    if not message.strip():
        return history, ""
    
    if not collection_name:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": "❌ 请先选择一个知识库"})
        return history, ""
    
    # 添加用户消息
    history.append({"role": "user", "content": message})
    yield history, ""
    
    # 添加空的助手消息用于流式更新
    history.append({"role": "assistant", "content": ""})
    yield history, ""
    
    # 格式化请求 - 转换消息格式
    old_format_history = []
    for i in range(0, len(history) - 2, 2):  # 排除最后的用户消息和空助手消息
        if i + 1 < len(history) - 2:
            user_msg = history[i]["content"] if history[i]["role"] == "user" else ""
            assistant_msg = history[i + 1]["content"] if history[i + 1]["role"] == "assistant" else ""
            old_format_history.append([user_msg, assistant_msg])
    
    payload = chatbot.format_messages_for_api(message, old_format_history, collection_name)
    
    # 流式获取回复
    full_response = ""
    for chunk in chatbot.query_rag_stream(payload):
        full_response += chunk
        history[-1]["content"] = full_response
        yield history, ""
    
    # 最终更新
    yield history, ""

# 自定义 CSS
custom_css = """
/* 设置全局字体 */
* {
    font-family: "Microsoft YaHei", "微软雅黑", "Segoe UI", Tahoma, Geneva, Verdana, sans-serif !important;
}

/* 美化标题 */
.gradio-container h1 {
    text-align: center;
    color: #2c3e50;
    margin-bottom: 2rem;
    font-weight: 600;
}

/* 美化选项卡 */
.tab-nav {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    border-radius: 10px 10px 0 0;
}

/* 美化按钮 */
.btn-primary {
    background: linear-gradient(45deg, #667eea, #764ba2);
    border: none;
    border-radius: 8px;
    color: white;
    font-weight: 500;
    transition: all 0.3s ease;
}

.btn-primary:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
}

/* 美化输入框 */
.gr-textbox {
    border-radius: 8px;
    border: 2px solid #e1e8ed;
    transition: border-color 0.3s ease;
}

.gr-textbox:focus {
    border-color: #667eea;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
}

/* 美化聊天区域 */
.chatbot {
    border-radius: 12px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

/* 状态消息样式 */
.status-success {
    color: #27ae60;
    font-weight: 500;
}

.status-error {
    color: #e74c3c;
    font-weight: 500;
}

/* 文件上传区域 */
.file-upload {
    border: 2px dashed #667eea;
    border-radius: 8px;
    padding: 2rem;
    text-align: center;
    transition: all 0.3s ease;
}

.file-upload:hover {
    border-color: #764ba2;
    background-color: #f8f9ff;
}
"""

# 创建 Gradio 界面
def create_interface():
    with gr.Blocks(css=custom_css, title="NVIDIA RAG 知识库管理系统") as demo:
        gr.Markdown("# 🤖 NVIDIA RAG 智能对话与知识库管理系统")
        
        with gr.Tabs():
            # 聊天选项卡
            with gr.Tab("💬 智能对话"):
                with gr.Row():
                    with gr.Column(scale=3):
                        chatbot_interface = gr.Chatbot(
                            label="对话区域",
                            height=500,
                            show_label=True,
                            show_copy_button=True,
                            type="messages"
                        )
                        
                        with gr.Row():
                            msg_input = gr.Textbox(
                                label="输入消息",
                                placeholder="请输入您的问题...",
                                scale=4
                            )
                            send_btn = gr.Button("发送", variant="primary", scale=1)
                    
                    with gr.Column(scale=1):
                        collection_selector = gr.Dropdown(
                            label="选择知识库",
                            choices=get_collections_list(),
                            value=current_collection,
                            interactive=True,
                            allow_custom_value=True
                        )
                        
                        refresh_btn = gr.Button("🔄 刷新知识库列表", variant="secondary")
                        
                        collection_status = gr.Textbox(
                            label="状态",
                            interactive=False,
                            lines=2
                        )
                
                # 绑定事件
                send_btn.click(
                    chat_fn,
                    inputs=[msg_input, chatbot_interface, collection_selector],
                    outputs=[chatbot_interface, msg_input]
                )
                
                msg_input.submit(
                    chat_fn,
                    inputs=[msg_input, chatbot_interface, collection_selector],
                    outputs=[chatbot_interface, msg_input]
                )
                
                collection_selector.change(
                    update_current_collection,
                    inputs=[collection_selector],
                    outputs=[collection_status]
                )
                
                refresh_btn.click(
                    refresh_collections,
                    outputs=[collection_selector, collection_selector, collection_selector]
                )
            
            # 知识库管理选项卡
            with gr.Tab("📚 知识库管理"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 🗂️ 知识库操作")
                        
                        with gr.Row():
                            new_collection_name = gr.Textbox(
                                label="新知识库名称",
                                placeholder="输入知识库名称..."
                            )
                            create_btn = gr.Button("➕ 创建知识库", variant="primary")
                        
                        with gr.Row():
                            collection_to_delete = gr.Dropdown(
                                label="选择要删除的知识库",
                                choices=get_collections_list(),
                                interactive=True,
                                allow_custom_value=True
                            )
                            delete_btn = gr.Button("🗑️ 删除知识库", variant="stop")
                        
                        kb_status = gr.Textbox(
                            label="操作状态",
                            interactive=False,
                            lines=3
                        )
                    
                    with gr.Column():
                        gr.Markdown("### 📄 文档管理")
                        
                        doc_collection_selector = gr.Dropdown(
                            label="选择知识库",
                            choices=get_collections_list(),
                            value=current_collection,
                            interactive=True,
                            allow_custom_value=True
                        )
                        
                        file_upload = gr.Files(
                            label="上传文档",
                            file_count="multiple",
                            file_types=[".pdf", ".txt", ".doc", ".docx", ".md"]
                        )
                        
                        upload_btn = gr.Button("📤 上传文档", variant="primary")
                        
                        documents_list = gr.CheckboxGroup(
                            label="知识库中的文档",
                            choices=[],
                            interactive=True
                        )
                        
                        with gr.Row():
                            refresh_docs_btn = gr.Button("🔄 刷新文档列表", variant="secondary")
                            delete_docs_btn = gr.Button("🗑️ 删除选中文档", variant="stop")
                        
                        doc_status = gr.Textbox(
                            label="文档操作状态",
                            interactive=False,
                            lines=3
                        )
                
                # 绑定知识库管理事件
                create_btn.click(
                    create_new_collection,
                    inputs=[new_collection_name],
                    outputs=[kb_status, collection_to_delete]
                )
                
                delete_btn.click(
                    delete_selected_collection,
                    inputs=[collection_to_delete],
                    outputs=[kb_status, collection_to_delete]
                )
                
                # 绑定文档管理事件
                doc_collection_selector.change(
                    refresh_documents,
                    inputs=[doc_collection_selector],
                    outputs=[documents_list]
                )
                
                upload_btn.click(
                    upload_files,
                    inputs=[file_upload, doc_collection_selector],
                    outputs=[doc_status, documents_list]
                )
                
                refresh_docs_btn.click(
                    refresh_documents,
                    inputs=[doc_collection_selector],
                    outputs=[documents_list]
                )
                
                delete_docs_btn.click(
                    delete_selected_documents,
                    inputs=[doc_collection_selector, documents_list],
                    outputs=[doc_status, documents_list]
                )
        
        # 页面加载时刷新数据
        demo.load(
            refresh_collections,
            outputs=[collection_selector, collection_to_delete, doc_collection_selector]
        )
    
    return demo

if __name__ == "__main__":
    # 创建并启动界面
    demo = create_interface()
    
    print("🚀 启动 NVIDIA RAG 知识库管理系统...")
    print(f"📡 RAG 服务器: {RAG_SERVER_URL}")
    print(f"📥 文档处理服务器: {INGESTOR_SERVER_URL}")
    print("🌐 访问地址: http://localhost:7860")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
