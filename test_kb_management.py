#!/usr/bin/env python3
"""
测试知识库管理功能
验证 Ingestor Server API 的各项功能
"""

import requests
import json
import os
import tempfile
import time

# 配置
INGESTOR_SERVER_URL = "http://192.168.81.253:8082"

def test_health_check():
    """测试健康检查"""
    print("🔍 测试 Ingestor Server 健康状态...")
    try:
        response = requests.get(f"{INGESTOR_SERVER_URL}/health")
        print(f"状态码: {response.status_code}")
        print(f"响应: {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ 健康检查失败: {e}")
        return False

def test_list_collections():
    """测试获取知识库列表"""
    print("\n📋 测试获取知识库列表...")
    try:
        response = requests.get(f"{INGESTOR_SERVER_URL}/collections")
        print(f"状态码: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"响应数据: {json.dumps(data, indent=2, ensure_ascii=False)}")
            return data
        else:
            print(f"❌ 请求失败: {response.text}")
            return None
    except Exception as e:
        print(f"❌ 获取知识库列表失败: {e}")
        return None

def test_create_collection(collection_name="test_kb_001"):
    """测试创建知识库"""
    print(f"\n➕ 测试创建知识库: {collection_name}")
    try:
        payload = {
            "collection_name": collection_name,
            "embedding_dimension": 2048,
            "metadata_schema": []
        }
        
        response = requests.post(
            f"{INGESTOR_SERVER_URL}/collection",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"状态码: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 创建成功: {json.dumps(data, indent=2, ensure_ascii=False)}")
            return True
        else:
            print(f"❌ 创建失败: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ 创建知识库失败: {e}")
        return False

def test_upload_document(collection_name="test_kb_001"):
    """测试上传文档"""
    print(f"\n📤 测试上传文档到知识库: {collection_name}")
    
    # 创建测试文档
    test_content = """
# 测试文档

这是一个测试文档，用于验证文档上传功能。

## 内容概述
- 测试知识库管理
- 验证文档处理能力
- 确保 API 正常工作

## 技术信息
- 文档格式: Markdown
- 编码: UTF-8
- 创建时间: """ + time.strftime("%Y-%m-%d %H:%M:%S")
    
    try:
        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
            f.write(test_content)
            temp_file = f.name
        
        # 上传文件
        with open(temp_file, 'rb') as f:
            files = {
                'documents': ('test_document.md', f, 'text/markdown')
            }
            
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
                f"{INGESTOR_SERVER_URL}/documents",
                files=files,
                data=data
            )
        
        # 清理临时文件
        os.unlink(temp_file)
        
        print(f"状态码: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 上传成功: {json.dumps(data, indent=2, ensure_ascii=False)}")
            return True
        else:
            print(f"❌ 上传失败: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ 上传文档失败: {e}")
        return False

def test_list_documents(collection_name="test_kb_001"):
    """测试获取文档列表"""
    print(f"\n📋 测试获取文档列表: {collection_name}")
    try:
        response = requests.get(
            f"{INGESTOR_SERVER_URL}/documents",
            params={"collection_name": collection_name}
        )
        
        print(f"状态码: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"文档列表: {json.dumps(data, indent=2, ensure_ascii=False)}")
            return data
        else:
            print(f"❌ 获取失败: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ 获取文档列表失败: {e}")
        return None

def test_delete_documents(collection_name="test_kb_001", document_names=None):
    """测试删除文档"""
    if document_names is None:
        # 先获取文档列表
        docs_data = test_list_documents(collection_name)
        if not docs_data:
            print("❌ 无法获取文档列表，跳过删除测试")
            return False
        
        # 提取文档名称
        if isinstance(docs_data, dict) and 'documents' in docs_data:
            documents = docs_data['documents']
        elif isinstance(docs_data, list):
            documents = docs_data
        else:
            print("❌ 文档数据格式不正确")
            return False
        
        if not documents:
            print("❌ 知识库中没有文档可删除")
            return False
        
        # 使用第一个文档进行测试
        if isinstance(documents[0], dict):
            document_names = [documents[0].get('name', documents[0].get('id', str(documents[0])))]
        else:
            document_names = [str(documents[0])]
    
    print(f"\n🗑️ 测试删除文档: {document_names}")
    try:
        response = requests.delete(
            f"{INGESTOR_SERVER_URL}/documents",
            params={"collection_name": collection_name},
            json=document_names,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"状态码: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 删除成功: {json.dumps(data, indent=2, ensure_ascii=False)}")
            return True
        else:
            print(f"❌ 删除失败: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ 删除文档失败: {e}")
        return False

def test_delete_collection(collection_name="test_kb_001"):
    """测试删除知识库"""
    print(f"\n🗑️ 测试删除知识库: {collection_name}")
    try:
        response = requests.delete(
            f"{INGESTOR_SERVER_URL}/collections",
            json=[collection_name],
            headers={"Content-Type": "application/json"}
        )
        
        print(f"状态码: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 删除成功: {json.dumps(data, indent=2, ensure_ascii=False)}")
            return True
        else:
            print(f"❌ 删除失败: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ 删除知识库失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🧪 开始测试知识库管理功能")
    print("=" * 50)
    
    # 1. 健康检查
    if not test_health_check():
        print("❌ Ingestor Server 不可用，停止测试")
        return
    
    # 2. 获取初始知识库列表
    initial_collections = test_list_collections()
    
    # 3. 创建测试知识库
    test_collection = "test_kb_" + str(int(time.time()))
    if not test_create_collection(test_collection):
        print("❌ 无法创建测试知识库，停止测试")
        return
    
    # 4. 验证知识库已创建
    updated_collections = test_list_collections()
    
    # 5. 上传测试文档
    if not test_upload_document(test_collection):
        print("⚠️ 文档上传失败，但继续测试其他功能")
    
    # 等待一段时间让文档处理完成
    print("\n⏳ 等待文档处理完成...")
    time.sleep(3)
    
    # 6. 获取文档列表
    test_list_documents(test_collection)
    
    # 7. 删除文档（如果有的话）
    # test_delete_documents(test_collection)
    
    # 8. 清理：删除测试知识库
    test_delete_collection(test_collection)
    
    # 9. 验证清理完成
    final_collections = test_list_collections()
    
    print("\n" + "=" * 50)
    print("🎉 测试完成！")

if __name__ == "__main__":
    main()
