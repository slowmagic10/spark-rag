#!/usr/bin/env python3
"""
增强版 NVIDIA RAG 知识库管理系统
特色功能：
- 文档向量化处理完成后才显示在列表中
- 实时处理状态跟踪
- 阻塞模式上传确保数据一致性
- 美化的进度指示器
"""

import gradio as gr
import requests
import json
import os
import time
import threading
from typing import List, Dict, Any, Optional, Tuple
import mimetypes
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 配置
RAG_SERVER_URL = "http://192.168.120.3:8081"
INGESTOR_SERVER_URL = "http://192.168.120.3:8082"

# 系统提示词，强制使用中文回复
SYSTEM_PROMPT = """
你是一个专业的AI助手。请严格遵循以下规则：
1. 必须使用中文回复所有问题
2. 回复要准确、有用、详细
3. 保持礼貌和专业的语气
4. 基于提供的文档内容进行回答
"""

class DocumentProcessor:
    """文档处理器 - 负责文档上传和状态跟踪"""
    
    def __init__(self, ingestor_url: str):
        self.ingestor_url = ingestor_url
        self.processing_tasks = {}  # 存储正在处理的任务
    
    def upload_document_blocking(self, file_path: str, collection_name: str, progress_callback=None) -> Tuple[bool, str]:
        """阻塞模式上传文档，确保处理完成"""
        try:
            file_name = os.path.basename(file_path)
            
            if progress_callback:
                progress_callback(f"📤 开始上传文档: {file_name}")
            
            # 准备文件
            with open(file_path, 'rb') as f:
                files = {
                    'documents': (file_name, f, self._get_mime_type(file_path))
                }
                
                # 使用阻塞模式
                data = {
                    'data': json.dumps({
                        "collection_name": collection_name,
                        "blocking": True,  # 关键：使用阻塞模式
                        "split_options": {
                            "chunk_size": 512,
                            "chunk_overlap": 150
                        },
                        "custom_metadata": [],
                        "generate_summary": False
                    })
                }
                
                if progress_callback:
                    progress_callback(f"🔄 正在处理文档: {file_name} (向量化中...)")
                
                start_time = time.time()
                response = requests.post(
                    f"{self.ingestor_url}/documents",
                    files=files,
                    data=data,
                    timeout=300  # 5分钟超时
                )
                
                processing_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                
                # 检查处理结果
                if data.get('failed_documents'):
                    failed_docs = data['failed_documents']
                    error_msg = f"文档处理失败: {failed_docs}"
                    if progress_callback:
                        progress_callback(f"❌ {error_msg}")
                    return False, error_msg
                
                success_msg = f"✅ 文档 {file_name} 处理完成 (耗时: {processing_time:.1f}秒)"
                if progress_callback:
                    progress_callback(success_msg)
                
                return True, success_msg
            else:
                error_msg = f"上传失败: {response.status_code} - {response.text}"
                if progress_callback:
                    progress_callback(f"❌ {error_msg}")
                return False, error_msg
                
        except requests.exceptions.Timeout:
            error_msg = f"文档处理超时 (>5分钟): {file_name}"
            if progress_callback:
                progress_callback(f"⏰ {error_msg}")
            return False, error_msg
        except Exception as e:
            error_msg = f"处理文档时发生错误: {str(e)}"
            if progress_callback:
                progress_callback(f"❌ {error_msg}")
            return False, error_msg
    
    def upload_document_async(self, file_path: str, collection_name: str, task_id: str, progress_callback=None) -> Tuple[bool, str]:
        """异步模式上传文档，带进度跟踪"""
        try:
            file_name = os.path.basename(file_path)
            
            if progress_callback:
                progress_callback(f"📤 开始上传文档: {file_name}")
            
            # 准备文件
            with open(file_path, 'rb') as f:
                files = {
                    'documents': (file_name, f, self._get_mime_type(file_path))
                }
                
                # 使用非阻塞模式
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
                api_task_id = data.get('task_id', 'unknown')
                
                # 存储任务信息
                self.processing_tasks[task_id] = {
                    'api_task_id': api_task_id,
                    'file_name': file_name,
                    'collection_name': collection_name,
                    'start_time': time.time(),
                    'status': 'processing'
                }
                
                if progress_callback:
                    progress_callback(f"🔄 文档已提交处理: {file_name} (任务ID: {api_task_id[:8]}...)")
                
                # 开始轮询检查状态
                self._poll_task_status(task_id, collection_name, file_name, progress_callback)
                
                return True, f"文档已开始处理: {file_name}"
            else:
                error_msg = f"上传失败: {response.status_code} - {response.text}"
                if progress_callback:
                    progress_callback(f"❌ {error_msg}")
                return False, error_msg
                
        except Exception as e:
            error_msg = f"上传文档时发生错误: {str(e)}"
            if progress_callback:
                progress_callback(f"❌ {error_msg}")
            return False, error_msg
    
    def _poll_task_status(self, task_id: str, collection_name: str, file_name: str, progress_callback=None):
        """轮询任务状态"""
        def poll():
            max_attempts = 60  # 最多轮询5分钟
            attempt = 0
            
            while attempt < max_attempts:
                try:
                    # 检查文档是否出现在列表中
                    response = requests.get(
                        f"{self.ingestor_url}/documents",
                        params={"collection_name": collection_name}
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        documents = data.get('documents', [])
                        
                        # 检查我们的文档是否在列表中
                        for doc in documents:
                            if doc.get('document_name') == file_name:
                                # 文档处理完成
                                if task_id in self.processing_tasks:
                                    elapsed = time.time() - self.processing_tasks[task_id]['start_time']
                                    self.processing_tasks[task_id]['status'] = 'completed'
                                    
                                    if progress_callback:
                                        progress_callback(f"✅ 文档处理完成: {file_name} (耗时: {elapsed:.1f}秒)")
                                
                                return
                    
                    # 更新进度
                    if progress_callback and attempt % 6 == 0:  # 每30秒更新一次
                        elapsed = time.time() - self.processing_tasks.get(task_id, {}).get('start_time', time.time())
                        progress_callback(f"🔄 处理中: {file_name} (已耗时: {elapsed:.0f}秒)")
                    
                    time.sleep(5)  # 每5秒检查一次
                    attempt += 1
                    
                except Exception as e:
                    logger.error(f"轮询任务状态时出错: {e}")
                    break
            
            # 超时处理
            if task_id in self.processing_tasks:
                self.processing_tasks[task_id]['status'] = 'timeout'
                if progress_callback:
                    progress_callback(f"⏰ 文档处理超时: {file_name}")
        
        # 在后台线程中运行轮询
        threading.Thread(target=poll, daemon=True).start()
    
    def _get_mime_type(self, file_path: str) -> str:
        """获取文件的 MIME 类型"""
        mime_type, _ = mimetypes.guess_type(file_path)
        return mime_type or 'application/octet-stream'


class KnowledgeBaseManager:
    """知识库管理器"""
    
    def __init__(self, ingestor_url: str):
        self.ingestor_url = ingestor_url
        self.doc_processor = DocumentProcessor(ingestor_url)
    
    def list_collections(self) -> List[str]:
        """获取所有知识库列表"""
        try:
            response = requests.get(f"{self.ingestor_url}/collections")
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, dict) and 'collections' in data:
                    collections = data['collections']
                    if isinstance(collections, list) and len(collections) > 0:
                        if isinstance(collections[0], dict):
                            return [col.get('collection_name', str(col)) for col in collections]
                        else:
                            return collections
                    return []
                elif isinstance(data, list):
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
        """获取指定知识库中的文档列表 - 只返回已完成处理的文档"""
        try:
            response = requests.get(
                f"{self.ingestor_url}/documents",
                params={"collection_name": collection_name}
            )
            
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, dict) and 'documents' in data:
                    documents = data['documents']
                    if isinstance(documents, list):
                        # 只返回已完成处理的文档名称
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
    
    def upload_documents_blocking(self, file_paths: List[str], collection_name: str, progress_callback=None) -> List[Tuple[str, bool, str]]:
        """批量上传文档 - 阻塞模式"""
        results = []
        
        for i, file_path in enumerate(file_paths):
            file_name = os.path.basename(file_path)
            
            if progress_callback:
                progress_callback(f"📊 处理进度: {i+1}/{len(file_paths)} - {file_name}")
            
            success, message = self.doc_processor.upload_document_blocking(
                file_path, collection_name, progress_callback
            )
            
            results.append((file_name, success, message))
        
        return results
    
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


class RAGChatBot:
    """RAG 聊天机器人 - 使用已验证的API格式"""
    
    def __init__(self, rag_server_url: str):
        self.rag_server_url = rag_server_url
        self.conversation_history = []
    
    def format_messages_for_api(self, new_message: str, chat_history: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Format messages for API call
        
        Args:
            new_message: The new user message
            chat_history: Previous conversation history in messages format
            
        Returns:
            Formatted messages list
        """
        # Start with system prompt
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        # Add existing conversation history (excluding the last empty assistant message)
        for msg in chat_history[:-1] if chat_history and chat_history[-1].get("content") == "" else chat_history:
            if msg.get("content"):  # Only add messages with content
                messages.append(msg)
        
        # Add the new user message
        messages.append({"role": "user", "content": new_message})
        
        return messages
    
    def query_rag_stream(self, messages: List[Dict[str, str]], 
                         collection_name: str = None,
                         temperature: float = 0.1, 
                         top_p: float = 0.9, 
                         max_tokens: int = 4096,
                         use_knowledge_base: bool = True):
        """
        Query the RAG API with streaming response
        """
        
        # 使用与前端相同的配置格式
        payload = {
            "messages": messages,
            "collection_names": [collection_name] if collection_name and use_knowledge_base else [],
            "temperature": temperature,
            "top_p": top_p,
            "reranker_top_k": 10,
            "vdb_top_k": 10,
            "use_knowledge_base": use_knowledge_base and bool(collection_name),
            "stream": True,
            "max_tokens": max_tokens
        }
        
        try:
            response = requests.post(
                f"{self.rag_server_url}/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"},
                stream=True,
                timeout=30
            )
            
            if response.status_code == 200:
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

def upload_files_enhanced(files, collection_name: str, progress_display):
    """增强的文件上传功能 - 统一使用阻塞模式，实时进度更新"""
    if not files:
        yield "❌ 请选择要上传的文件", gr.update(), "❌ 请选择要上传的文件"
        return
    
    if not collection_name:
        yield "❌ 请选择目标知识库", gr.update(), "❌ 请选择目标知识库"
        return
    
    # 开始处理
    yield "", gr.update(), "🚀 开始文档上传和处理..."
    
    file_paths = [file.name for file in files]
    results = []
    
    for i, file_path in enumerate(file_paths):
        file_name = os.path.basename(file_path)
        
        # 显示当前处理进度
        progress_msg = f"📊 处理进度: {i+1}/{len(file_paths)} - {file_name}\n🔄 正在上传和向量化处理中..."
        yield "", gr.update(), progress_msg
        
        # 执行上传
        success, message = kb_manager.doc_processor.upload_document_blocking(
            file_path, collection_name
        )
        
        results.append((file_name, success, message))
        
        # 更新单个文件完成状态
        status_icon = "✅" if success else "❌"
        complete_msg = f"📊 处理进度: {i+1}/{len(file_paths)}\n{status_icon} {file_name}: {message}"
        yield "", gr.update(), complete_msg
    
    # 处理结果
    success_count = sum(1 for _, success, _ in results if success)
    failed_count = len(results) - success_count
    
    final_status = []
    for file_name, success, message in results:
        status_icon = "✅" if success else "❌"
        final_status.append(f"{status_icon} {file_name}: {message}")
    
    # 上传完成后刷新文档列表
    updated_documents = get_documents_list(collection_name)
    
    summary = f"📊 上传完成: 成功 {success_count} 个，失败 {failed_count} 个\n\n" + "\n".join(final_status)
    progress_final = f"🎉 全部完成！成功: {success_count}, 失败: {failed_count}\n\n详细结果:\n" + "\n".join(final_status)
    
    yield summary, gr.update(choices=updated_documents), progress_final

def delete_selected_documents(collection_name: str, selected_documents: List[str]):
    """删除选中的文档"""
    if not collection_name:
        return "❌ 请选择知识库", gr.update()
    
    if not selected_documents:
        return "❌ 请选择要删除的文档", gr.update()
    
    success, message = kb_manager.delete_documents(collection_name, selected_documents)
    
    if success:
        updated_documents = get_documents_list(collection_name)
        return f"✅ {message}", gr.update(choices=updated_documents, value=[])
    else:
        return f"❌ {message}", gr.update()

def update_current_collection(collection_name: str):
    """更新当前选中的知识库"""
    global current_collection
    current_collection = collection_name
    return f"✅ 已切换到知识库: {collection_name}"

def chat_fn(message: str, history: List[List[str]], collection_name: str, 
             use_kb: bool, temperature: float, 
             top_p: float, max_tokens: int):
    """聊天函数 - 使用元组格式，支持参数控制"""
    if not message.strip():
        return history, ""
    
    # 如果启用知识库但没有选择知识库，则提示
    if use_kb and not collection_name:
        history.append([message, "❌ 已启用知识库但未选择知识库，请先选择一个知识库或关闭知识库功能"])
        return history, ""
    
    # 立即显示用户消息
    history.append([message, ""])
    yield history, ""
    
    # 从元组历史转换为消息格式用于API调用
    api_messages = []
    for user_msg, assistant_msg in history[:-1]:  # 排除当前空的assistant消息
        api_messages.append({"role": "user", "content": user_msg})
        if assistant_msg:
            api_messages.append({"role": "assistant", "content": assistant_msg})
    
    # 格式化消息
    messages = chatbot.format_messages_for_api(message, api_messages)
    
    # 流式获取回复
    full_response = ""
    for chunk in chatbot.query_rag_stream(
        messages=messages,
        collection_name=collection_name if use_kb else None,
        use_knowledge_base=use_kb,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens
    ):
        full_response += chunk
        history[-1] = [message, full_response]
        yield history, ""
    
    yield history, ""

def reset_parameters():
    """重置参数到默认值"""
    return True, 0.1, 0.9, 1024

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

/* 进度指示器样式 */
.progress-box {
    background: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 8px;
    padding: 1rem;
    font-family: 'Monaco', 'Menlo', monospace;
    font-size: 0.9em;
    max-height: 200px;
    overflow-y: auto;
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

.status-processing {
    color: #f39c12;
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
        gr.Markdown("# 🤖 NVIDIA RAG 知识库管理系统")
        
        with gr.Tabs():
            # 聊天选项卡
            with gr.Tab("💬 智能对话"):
                with gr.Row():
                    with gr.Column(scale=4):  # 增加聊天区域比例
                        chatbot_interface = gr.Chatbot(
                            label="对话区域",
                            height=600,  # 增加高度
                            show_label=True,
                            show_copy_button=True
                            # 去掉 type="messages"，使用默认的元组格式
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
                        
                        gr.Markdown("---")
                        gr.Markdown("### 生成参数")
                        
                        use_kb = gr.Checkbox(
                            label="启用知识库",
                            value=True,
                            info="基于向量数据库内容生成回答"
                        )
                        
                        temperature = gr.Slider(
                            label="随机性",
                            minimum=0.0,
                            maximum=1.0,
                            value=0.1,
                            step=0.1,
                            info="控制回答的随机性"
                        )
                        
                        top_p = gr.Slider(
                            label="核采样",
                            minimum=0.1,
                            maximum=1.0,
                            value=0.9,
                            step=0.1,
                            info="控制词汇选择的多样性"
                        )
                        
                        max_tokens = gr.Slider(
                            label="最大长度",
                            minimum=256,
                            maximum=4096,
                            value=1024,
                            step=256,
                            info="生成回答的最大字符数"
                        )
                        
                        reset_params_btn = gr.Button(
                            "🔄 重置参数", 
                            variant="secondary", 
                            size="sm"
                        )
                
                # 绑定事件
                send_btn.click(
                    chat_fn,
                    inputs=[msg_input, chatbot_interface, collection_selector, 
                           use_kb, temperature, top_p, max_tokens],
                    outputs=[chatbot_interface, msg_input]
                )
                
                msg_input.submit(
                    chat_fn,
                    inputs=[msg_input, chatbot_interface, collection_selector,
                           use_kb, temperature, top_p, max_tokens],
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
                
                reset_params_btn.click(
                    reset_parameters,
                    outputs=[use_kb, temperature, top_p, max_tokens]
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
                        
                        # 处理进度显示
                        progress_display = gr.Textbox(
                            label="📊 处理进度",
                            interactive=False,
                            lines=6,
                            elem_classes=["progress-box"]
                        )
                        
                        documents_list = gr.CheckboxGroup(
                            label="已完成处理的文档",
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
                    upload_files_enhanced,
                    inputs=[file_upload, doc_collection_selector, progress_display],
                    outputs=[doc_status, documents_list, progress_display]
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
