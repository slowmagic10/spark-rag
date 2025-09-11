#!/usr/bin/env python3
"""
支持热重载的 Gradio 应用启动器
使用 watchdog 监控文件变化，自动重启应用
"""

import os
import sys
import time
import subprocess
import signal
from pathlib import Path

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
except ImportError:
    print("⚠️  未安装 watchdog，正在安装...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "watchdog"])
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler

class GradioReloadHandler(FileSystemEventHandler):
    def __init__(self, restart_callback):
        self.restart_callback = restart_callback
        self.last_modified = {}
        
    def on_modified(self, event):
        if event.is_directory:
            return
            
        # 只监控 Python 文件
        if not event.src_path.endswith('.py'):
            return
            
        # 避免重复触发
        current_time = time.time()
        if event.src_path in self.last_modified:
            if current_time - self.last_modified[event.src_path] < 1:
                return
                
        self.last_modified[event.src_path] = current_time
        
        print(f"📝 检测到文件变化: {event.src_path}")
        print("🔄 正在重启 Gradio 应用...")
        self.restart_callback()

class GradioHotReloader:
    def __init__(self, script_path):
        self.script_path = script_path
        self.process = None
        self.observer = None
        
    def start_gradio(self):
        """启动 Gradio 应用"""
        if self.process:
            self.stop_gradio()
            
        print(f"🚀 启动 Gradio 应用: {self.script_path}")
        self.process = subprocess.Popen([
            sys.executable, self.script_path
        ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        # 实时输出日志
        def output_reader():
            for line in iter(self.process.stdout.readline, ''):
                print(line.rstrip())
                
        import threading
        threading.Thread(target=output_reader, daemon=True).start()
        
    def stop_gradio(self):
        """停止 Gradio 应用"""
        if self.process:
            print("⏹️  停止当前应用...")
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None
            time.sleep(1)  # 等待端口释放
            
    def restart_gradio(self):
        """重启 Gradio 应用"""
        self.start_gradio()
        
    def start_watching(self):
        """开始监控文件变化"""
        event_handler = GradioReloadHandler(self.restart_gradio)
        self.observer = Observer()
        
        # 监控当前目录
        watch_path = Path(self.script_path).parent
        self.observer.schedule(event_handler, str(watch_path), recursive=False)
        
        print(f"👀 开始监控目录: {watch_path}")
        print("💡 修改 Python 文件后将自动重启应用")
        print("🔥 热重载模式已启用")
        print("-" * 50)
        
        self.observer.start()
        
    def stop_watching(self):
        """停止监控"""
        if self.observer:
            self.observer.stop()
            self.observer.join()
            
    def run(self):
        """运行热重载器"""
        try:
            self.start_gradio()
            self.start_watching()
            
            # 保持运行
            while True:
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n🛑 收到停止信号...")
        finally:
            self.stop_watching()
            self.stop_gradio()
            print("👋 热重载器已停止")

def main():
    script_path = "gradio_chat_app.py"
    
    if not os.path.exists(script_path):
        print(f"❌ 找不到脚本文件: {script_path}")
        sys.exit(1)
        
    print("🔥 NVIDIA RAG Gradio 热重载启动器")
    print("=" * 50)
    
    reloader = GradioHotReloader(script_path)
    reloader.run()

if __name__ == "__main__":
    main()
