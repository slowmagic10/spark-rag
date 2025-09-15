#!/usr/bin/env python3
"""
测试文档上传和处理流程
验证文档向量化处理的时机
"""

import requests
import json
import os
import tempfile
import time

# 配置
INGESTOR_SERVER_URL = "http://192.168.81.253:8082"

def test_upload_with_blocking(collection_name="test", blocking=True):
    """测试阻塞模式上传文档"""
    print(f"\n📤 测试上传文档 (blocking={blocking}) 到知识库: {collection_name}")
    
    # 创建测试文档
    test_content = f"""
# 测试文档 - {time.strftime("%Y-%m-%d %H:%M:%S")}

这是一个测试文档，用于验证文档上传和处理流程。

## 测试内容
- 测试blocking模式: {blocking}
- 时间戳: {time.time()}
- 向量化处理测试

## 详细信息
本文档用于测试NVIDIA RAG系统的文档处理流程，
特别是测试向量化处理完成后文档才显示在列表中的功能。
"""
    
    try:
        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
            f.write(test_content)
            temp_file = f.name
        
        print(f"📝 创建的测试文件: {os.path.basename(temp_file)}")
        
        # 上传前检查文档列表
        print("\n📋 上传前文档列表:")
        docs_before = get_documents_list(collection_name)
        print(f"文档数量: {len(docs_before)}")
        
        start_time = time.time()
        
        # 上传文件
        with open(temp_file, 'rb') as f:
            files = {
                'documents': (f'test_doc_{int(time.time())}.md', f, 'text/markdown')
            }
            
            data = {
                'data': json.dumps({
                    "collection_name": collection_name,
                    "blocking": blocking,  # 关键参数
                    "split_options": {
                        "chunk_size": 512,
                        "chunk_overlap": 150
                    },
                    "custom_metadata": [],
                    "generate_summary": False
                })
            }
            
            print(f"\n🚀 开始上传 (blocking={blocking})...")
            response = requests.post(
                f"{INGESTOR_SERVER_URL}/documents",
                files=files,
                data=data
            )
        
        upload_time = time.time() - start_time
        
        # 清理临时文件
        os.unlink(temp_file)
        
        print(f"上传耗时: {upload_time:.2f}秒")
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 上传响应: {json.dumps(data, indent=2, ensure_ascii=False)}")
            
            # 立即检查文档列表
            print("\n📋 上传后立即检查文档列表:")
            docs_immediately = get_documents_list(collection_name)
            print(f"文档数量: {len(docs_immediately)}")
            
            if not blocking:
                # 对于非阻塞模式，等待一段时间后再检查
                wait_times = [1, 3, 5, 10]
                for wait_time in wait_times:
                    print(f"\n⏳ 等待 {wait_time} 秒后检查...")
                    time.sleep(wait_time)
                    docs_after_wait = get_documents_list(collection_name)
                    print(f"文档数量: {len(docs_after_wait)}")
                    if len(docs_after_wait) > len(docs_before):
                        print(f"✅ 文档在 {wait_time} 秒后出现在列表中")
                        break
                else:
                    print("⚠️ 等待10秒后文档仍未出现在列表中")
            
            return True
        else:
            print(f"❌ 上传失败: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ 上传文档失败: {e}")
        return False

def get_documents_list(collection_name):
    """获取文档列表"""
    try:
        response = requests.get(
            f"{INGESTOR_SERVER_URL}/documents",
            params={"collection_name": collection_name}
        )
        
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, dict) and 'documents' in data:
                documents = data['documents']
                print(f"文档详情: {json.dumps(documents, indent=2, ensure_ascii=False)}")
                return documents
            return []
        else:
            print(f"❌ 获取文档列表失败: {response.status_code} - {response.text}")
            return []
            
    except Exception as e:
        print(f"❌ 获取文档列表错误: {e}")
        return []

def main():
    """主测试函数"""
    print("🧪 开始测试文档上传和处理流程")
    print("=" * 50)
    
    test_collection = "test"
    
    # 1. 测试非阻塞模式 (当前默认)
    print("\n🔸 测试1: 非阻塞模式上传")
    test_upload_with_blocking(test_collection, blocking=False)
    
    # 2. 测试阻塞模式
    print("\n🔸 测试2: 阻塞模式上传")
    test_upload_with_blocking(test_collection, blocking=True)
    
    print("\n" + "=" * 50)
    print("🎉 测试完成！")

if __name__ == "__main__":
    main()
