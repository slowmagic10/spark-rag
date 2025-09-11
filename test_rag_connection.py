#!/usr/bin/env python3
"""
测试远程 NVIDIA RAG 服务器连接
"""

import requests
import json
import time

# 远程服务器配置
SERVER_IP = "192.168.81.253"
SERVER_PORT = "8081"
BASE_URL = f"http://{SERVER_IP}:{SERVER_PORT}/v1"

def test_health_check():
    """测试服务器健康状态"""
    print(f"🔍 测试服务器健康状态: {BASE_URL}/health")
    
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        print(f"   状态码: {response.status_code}")
        
        if response.status_code == 200:
            print("   ✅ 服务器健康状态正常")
            try:
                health_data = response.json()
                print(f"   📊 健康数据: {json.dumps(health_data, indent=2, ensure_ascii=False)}")
                
                # 检查依赖服务状态
                if 'check_dependencies' not in str(response.url):
                    print("   🔍 检查依赖服务状态...")
                    dep_response = requests.get(f"{BASE_URL}/health?check_dependencies=true", timeout=15)
                    if dep_response.status_code == 200:
                        dep_data = dep_response.json()
                        print(f"   📊 依赖服务状态: {json.dumps(dep_data, indent=2, ensure_ascii=False)}")
                    else:
                        print(f"   ⚠️  依赖服务检查失败: {dep_response.status_code}")
                        
            except:
                print(f"   📄 响应内容: {response.text}")
            return True
        else:
            print(f"   ❌ 服务器返回错误状态码: {response.status_code}")
            print(f"   📄 错误内容: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"   ❌ 连接错误: 无法连接到 {SERVER_IP}:{SERVER_PORT}")
        return False
    except requests.exceptions.Timeout:
        print(f"   ❌ 超时错误: 服务器响应超时")
        return False
    except Exception as e:
        print(f"   ❌ 其他错误: {str(e)}")
        return False

def test_simple_query():
    """测试简单的查询请求"""
    print(f"\n🤖 测试简单查询: {BASE_URL}/generate")
    
    # 尝试与前端相同的请求格式
    test_payloads = [
        # 配置1: 简化版本，类似前端
        {
            "messages": [{"role": "user", "content": "你好"}],
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
        # 配置2: 最简版本
        {
            "messages": [{"role": "user", "content": "Hello"}],
            "use_knowledge_base": False,
            "temperature": 0.1,
            "max_tokens": 100
        },
        # 配置3: 使用知识库
        {
            "messages": [{"role": "user", "content": "你好"}],
            "collection_names": ["test"],
            "temperature": 0.1,
            "top_p": 0.9,
            "reranker_top_k": 10,
            "vdb_top_k": 10,
            "confidence_threshold": 0.5,
            "use_knowledge_base": True,
            "enable_citations": True,
            "enable_guardrails": False
        }
    ]
    
    for i, test_payload in enumerate(test_payloads, 1):
        print(f"\n   📤 测试 {i}: 发送请求...")
        print(f"   📄 请求内容: {json.dumps(test_payload, indent=2, ensure_ascii=False)}")
        
        try:
            response = requests.post(
                f"{BASE_URL}/generate",
                json=test_payload,
                headers={"Content-Type": "application/json"},
                timeout=60
            )
            
            print(f"   状态码: {response.status_code}")
            
            if response.status_code == 200:
                print(f"   ✅ 测试 {i} 成功")
                try:
                    # 尝试解析 JSON 响应
                    result = response.json()
                    print(f"   📊 响应数据结构: {json.dumps({k: type(v).__name__ for k, v in result.items()}, indent=2)}")
                    
                    if "choices" in result and len(result["choices"]) > 0:
                        content = result["choices"][0].get("message", {}).get("content", "")
                        print(f"   💬 AI回复: {content[:200]}{'...' if len(content) > 200 else ''}")
                    else:
                        print(f"   ⚠️  响应格式异常: {json.dumps(result, indent=2, ensure_ascii=False)}")
                        
                except json.JSONDecodeError:
                    # 可能是流式响应
                    print(f"   📄 响应内容 (可能是流式): {response.text[:300]}...")
                    
                return True
            else:
                print(f"   ❌ 测试 {i} 失败: {response.status_code}")
                error_text = response.text
                print(f"   📄 错误内容: {error_text[:300]}{'...' if len(error_text) > 300 else ''}")
                # 继续尝试下一个测试
                
        except requests.exceptions.ConnectionError:
            print(f"   ❌ 连接错误: 无法连接到服务器")
            return False
        except requests.exceptions.Timeout:
            print(f"   ❌ 超时错误: 请求超时（可能是模型推理时间较长）")
            return False
        except Exception as e:
            print(f"   ❌ 其他错误: {str(e)}")
            # 继续尝试下一个测试
    
    return False

def test_chat_completions():
    """测试 OpenAI 兼容的 chat/completions 接口"""
    print(f"\n💬 测试 Chat Completions 接口: {BASE_URL}/chat/completions")
    
    test_payload = {
        "messages": [
            {"role": "user", "content": "什么是人工智能？"}
        ],
        "use_knowledge_base": True,
        "temperature": 0.1,
        "max_tokens": 500
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/chat/completions",
            json=test_payload,
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        
        print(f"   状态码: {response.status_code}")
        
        if response.status_code == 200:
            print("   ✅ Chat Completions 接口正常")
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0].get("message", {}).get("content", "")
                print(f"   💬 AI回复: {content[:200]}{'...' if len(content) > 200 else ''}")
            return True
        else:
            print(f"   ❌ Chat Completions 接口错误: {response.status_code}")
            print(f"   📄 错误内容: {response.text}")
            return False
            
    except Exception as e:
        print(f"   ❌ 错误: {str(e)}")
        return False

def test_network_connectivity():
    """测试基本网络连通性"""
    print(f"🌐 测试网络连通性...")
    
    # 测试 ping
    import subprocess
    try:
        result = subprocess.run(['ping', '-c', '3', SERVER_IP], 
                              capture_output=True, text=True, timeout=15)
        if result.returncode == 0:
            print(f"   ✅ 网络连通正常 (ping {SERVER_IP})")
        else:
            print(f"   ❌ 网络连通失败 (ping {SERVER_IP})")
            print(f"   📄 错误: {result.stderr}")
    except subprocess.TimeoutExpired:
        print(f"   ⚠️  Ping 超时")
    except Exception as e:
        print(f"   ⚠️  Ping 测试失败: {str(e)}")
    
    # 测试端口连通性
    import socket
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((SERVER_IP, int(SERVER_PORT)))
        sock.close()
        
        if result == 0:
            print(f"   ✅ 端口 {SERVER_PORT} 连通正常")
        else:
            print(f"   ❌ 端口 {SERVER_PORT} 连接失败")
    except Exception as e:
        print(f"   ❌ 端口测试失败: {str(e)}")

def main():
    """主函数：执行所有测试"""
    print("🚀 开始测试远程 NVIDIA RAG 服务器连接")
    print(f"🎯 目标服务器: {SERVER_IP}:{SERVER_PORT}")
    print("=" * 60)
    
    # 测试网络连通性
    test_network_connectivity()
    print()
    
    # 测试服务器健康状态
    health_ok = test_health_check()
    
    if health_ok:
        # 测试查询接口
        test_simple_query()
        
        # 测试 Chat Completions 接口
        test_chat_completions()
    else:
        print("\n⚠️  由于健康检查失败，跳过其他API测试")
    
    print("\n" + "=" * 60)
    print("🏁 测试完成")
    
    if health_ok:
        print("✅ 服务器连接正常，可以启动 Gradio 聊天应用")
        print(f"🚀 运行命令: python3 gradio_chat_app.py")
    else:
        print("❌ 服务器连接异常，请检查:")
        print("   1. 服务器是否正在运行")
        print("   2. 防火墙设置")
        print("   3. 网络连接")
        print("   4. 服务器地址和端口是否正确")

if __name__ == "__main__":
    main()
