#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Azure GPT 图片分析对话系统启动脚本
"""

import sys
import os

def check_dependencies():
    """检查依赖包是否安装"""
    required_packages = [
        'flask',
        'flask-cors',
        'litellm',
        'pillow',
        'python-dotenv',
        'requests',
        'werkzeug'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ 缺少以下依赖包:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n请运行以下命令安装依赖:")
        print("pip install -r requirements.txt")
        return False
    
    return True

def check_config():
    """检查配置文件"""
    from config import Config

    missing = []
    if not Config.API_KEY or Config.API_KEY == "your-api-key-here":
        missing.append("api_key")
    if not Config.API_MODEL:
        missing.append("api_model")
    if Config.API_PROVIDER in {"openai_compatible", "others"} and not Config.API_ENDPOINT:
        missing.append("api_endpoint")

    if missing:
        print("❌ 缺少必要配置（可来自环境变量/.env/pipeline_config.yaml）:")
        for var in missing:
            print(f"   - {var}")
        return False

    print(f"✅ pic_analyze provider: {Config.API_PROVIDER or 'openai'}")
    
    return True

def main():
    """主函数"""
    print("🚀 启动Azure GPT图片分析对话系统...")
    print("=" * 50)
    
    # 检查依赖
    print("📦 检查依赖包...")
    if not check_dependencies():
        sys.exit(1)
    print("✅ 依赖包检查通过")
    
    # 检查配置
    print("⚙️  检查配置文件...")
    if not check_config():
        sys.exit(1)
    print("✅ 配置文件检查通过")
    
    # 创建必要的文件夹
    print("📁 创建必要文件夹...")
    os.makedirs('uploads', exist_ok=True)
    print("✅ 文件夹创建完成")
    
    print("\n🎉 系统准备就绪!")
    print("=" * 50)
    
    # 启动自动聚类分析系统
    try:
        from auto_analyzer import main as auto_analysis_main
        auto_analysis_main()
    except KeyboardInterrupt:
        print("\n\n👋 程序被用户中断，再见！")
    except Exception as e:
        print(f"\n❌ 程序运行出错: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
