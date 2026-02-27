#!/bin/bash

echo "启动Azure GPT图片分析对话系统..."
echo

# 检查Python是否安装
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到Python3，请先安装Python 3.8+"
    exit 1
fi

# 检查依赖是否安装
echo "检查依赖包..."
if ! python3 -c "import openai" &> /dev/null; then
    echo "安装依赖包..."
    pip3 install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "错误: 依赖包安装失败"
        exit 1
    fi
fi

# 检查配置文件
if [ ! -f .env ]; then
    echo "错误: 配置文件不存在"
    echo "请复制 env_example.txt 为 .env 并填入您的Azure OpenAI配置"
    exit 1
fi

# 运行系统
echo "启动系统..."
python3 run.py

