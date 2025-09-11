#!/bin/bash

# NVIDIA RAG Gradio Chat App 启动脚本

echo "🚀 NVIDIA RAG Gradio 聊天应用启动器"
echo "选择启动模式:"
echo "1) 普通模式"
echo "2) 热重载模式 (开发用)"
echo ""

read -p "请选择模式 (1-2): " mode

case $mode in
    1)
        echo "📍 启动普通模式..."
        ;;
    2)
        echo "🔥 启动热重载模式..."
        echo "💡 提示: 修改 gradio_chat_app.py 后应用将自动重启"
        ;;
    *)
        echo "❌ 无效选择，使用普通模式"
        mode=1
        ;;
esac

echo ""

# 检查 Python 版本
python_version=$(python3 --version 2>&1 | grep -o '[0-9]\+\.[0-9]\+' | head -1)
echo "📍 Python 版本: $python_version"

# 检查并安装依赖
echo "📦 检查并安装依赖..."
if command -v pip3 &> /dev/null; then
    pip3 install -r gradio_requirements.txt > /dev/null 2>&1
elif command -v pip &> /dev/null; then
    pip install -r gradio_requirements.txt > /dev/null 2>&1
else
    echo "❌ 错误: 未找到 pip，请先安装 Python 和 pip"
    exit 1
fi

# 检查 RAG 服务器状态
echo "🔍 检查 RAG 服务器状态..."
if curl -s http://192.168.81.253:8081/v1/health > /dev/null; then
    echo "✅ RAG 服务器运行正常"
else
    echo "⚠️  警告: RAG 服务器未响应 (http://192.168.81.253:8081)"
    echo "   请确保先启动 RAG 服务器"
fi

echo ""

# 根据选择启动对应模式
if [ "$mode" = "2" ]; then
    echo "🌐 启动热重载模式..."
    python3 gradio_simple_reload.py
else
    echo "🌐 启动普通模式..."
    python3 gradio_chat_app.py
fi
