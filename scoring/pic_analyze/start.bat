@echo off
echo 启动Azure GPT图片分析对话系统...
echo.

REM 检查Python是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到Python，请先安装Python 3.8+
    pause
    exit /b 1
)

REM 检查依赖是否安装
echo 检查依赖包...
pip show openai >nul 2>&1
if errorlevel 1 (
    echo 安装依赖包...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo 错误: 依赖包安装失败
        pause
        exit /b 1
    )
)

REM 检查配置文件
if not exist .env (
    echo 错误: 配置文件不存在
    echo 请复制 env_example.txt 为 .env 并填入您的Azure OpenAI配置
    pause
    exit /b 1
)

REM 运行系统
echo 启动系统...
python run.py

pause

