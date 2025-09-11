#!/usr/bin/env python3
"""
简化版自动重载启动器
"""

import os
import sys
import time
import subprocess
import signal
from pathlib import Path

class SimpleReloader:
    def __init__(self, script_path):
        self.script_path = script_path
        self.process = None
        self.last_modified = 0
        
    def get_file_mtime(self):
        """获取文件最后修改时间"""
        try:
            return os.path.getmtime(self.script_path)
        except OSError:
            return 0
            
    def start_gradio(self):
        """启动 Gradio 应用"""
        if self.process:
            self.stop_gradio()
            
        print(f"🚀 启动 Gradio 应用: {self.script_path}")
        env = os.environ.copy()
        env['GRADIO_RELOAD'] = 'true'  # 设置环境变量标识重载模式
        
        self.process = subprocess.Popen([
            sys.executable, self.script_path
        ], env=env)
        
    def stop_gradio(self):
        """停止 Gradio 应用"""
        if self.process:
            print("⏹️  停止当前应用...")
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None
            time.sleep(1)  # 等待端口释放
            
    def check_and_reload(self):
        """检查文件是否有变化并重载"""
        current_mtime = self.get_file_mtime()
        
        if current_mtime > self.last_modified:
            if self.last_modified > 0:  # 跳过初始启动
                print(f"📝 检测到文件变化: {self.script_path}")
                print("🔄 正在重启应用...")
                self.start_gradio()
            else:
                print(f"👀 开始监控文件: {self.script_path}")
                
            self.last_modified = current_mtime
            
    def run(self):
        """运行重载器"""
        print("🔥 NVIDIA RAG Gradio 简化重载器")
        print("💡 修改 gradio_chat_app.py 后将自动重启")
        print("⌨️  按 Ctrl+C 停止")
        print("-" * 50)
        
        try:
            self.start_gradio()
            self.last_modified = self.get_file_mtime()
            
            while True:
                time.sleep(1)
                self.check_and_reload()
                
                # 检查进程是否还在运行
                if self.process and self.process.poll() is not None:
                    print("⚠️  应用意外退出，正在重启...")
                    self.start_gradio()
                    
        except KeyboardInterrupt:
            print("\n🛑 收到停止信号...")
        finally:
            self.stop_gradio()
            print("👋 重载器已停止")

def main():
    script_path = "gradio_chat_app.py"
    
    if not os.path.exists(script_path):
        print(f"❌ 找不到脚本文件: {script_path}")
        sys.exit(1)
        
    reloader = SimpleReloader(script_path)
    reloader.run()

if __name__ == "__main__":
    main()
