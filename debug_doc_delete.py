#!/usr/bin/env python3
"""
调试文档删除功能的脚本
"""

import requests
import json

INGESTOR_SERVER_URL = "http://192.168.81.253:8082"

def debug_list_documents(collection_name="test"):
    """调试获取文档列表"""
    print(f"🔍 调试获取文档列表: {collection_name}")
    try:
        response = requests.get(
            f"{INGESTOR_SERVER_URL}/documents",
            params={"collection_name": collection_name}
        )
        
        print(f"状态码: {response.status_code}")
        print(f"响应头: {dict(response.headers)}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"响应数据类型: {type(data)}")
            print(f"响应数据: {json.dumps(data, indent=2, ensure_ascii=False)}")
            
            # 分析数据结构
            if isinstance(data, dict):
                print("📋 数据是字典格式")
                for key, value in data.items():
                    print(f"  - {key}: {type(value)}")
                    if key == 'documents' and isinstance(value, list) and value:
                        print(f"    第一个文档: {value[0]}")
                        print(f"    文档数量: {len(value)}")
            elif isinstance(data, list):
                print("📋 数据是列表格式")
                print(f"  文档数量: {len(data)}")
                if data:
                    print(f"  第一个文档: {data[0]}")
            
            return data
        else:
            print(f"❌ 请求失败: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ 获取文档列表失败: {e}")
        return None

def debug_delete_documents(collection_name="test", document_names=None):
    """调试删除文档"""
    if document_names is None:
        # 先获取文档列表
        docs_data = debug_list_documents(collection_name)
        if not docs_data:
            print("❌ 无法获取文档列表")
            return False
        
        # 提取文档名称进行测试
        if isinstance(docs_data, dict) and 'documents' in docs_data:
            documents = docs_data['documents']
        elif isinstance(docs_data, list):
            documents = docs_data
        else:
            print("❌ 无法解析文档数据")
            return False
        
        if not documents:
            print("❌ 没有文档可删除")
            return False
        
        # 选择第一个文档进行删除测试
        first_doc = documents[0]
        if isinstance(first_doc, dict):
            # 尝试不同的字段名
            doc_name = (first_doc.get('document_name') or 
                       first_doc.get('name') or 
                       first_doc.get('filename') or 
                       first_doc.get('id') or 
                       str(first_doc))
        else:
            doc_name = str(first_doc)
        
        document_names = [doc_name]
    
    print(f"\n🗑️ 调试删除文档: {document_names} from {collection_name}")
    
    try:
        # 构建删除请求
        url = f"{INGESTOR_SERVER_URL}/documents"
        params = {"collection_name": collection_name}
        headers = {"Content-Type": "application/json"}
        payload = document_names
        
        print(f"请求 URL: {url}")
        print(f"请求参数: {params}")
        print(f"请求头: {headers}")
        print(f"请求体: {json.dumps(payload, ensure_ascii=False)}")
        
        response = requests.delete(
            url,
            params=params,
            json=payload,
            headers=headers
        )
        
        print(f"响应状态码: {response.status_code}")
        print(f"响应头: {dict(response.headers)}")
        print(f"响应文本: {response.text}")
        
        if response.status_code == 200:
            try:
                data = response.json()
                print(f"响应数据: {json.dumps(data, indent=2, ensure_ascii=False)}")
            except:
                print("响应不是有效的 JSON")
            
            print("✅ 删除请求成功")
            
            # 再次获取文档列表验证删除结果
            print("\n🔍 删除后验证文档列表:")
            debug_list_documents(collection_name)
            
            return True
        else:
            print(f"❌ 删除失败: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ 删除文档时发生错误: {e}")
        return False

if __name__ == "__main__":
    print("🧪 开始调试文档删除功能")
    print("=" * 50)
    
    # 使用 test 知识库进行调试
    collection_name = "test"
    
    # 1. 先查看文档列表
    print("第一步：查看文档列表")
    debug_list_documents(collection_name)
    
    # 2. 尝试删除文档
    print("\n第二步：尝试删除文档")
    debug_delete_documents(collection_name)
