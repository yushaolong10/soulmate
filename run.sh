
# 启动脚本
echo "Starting..."

# 检查虚拟环境
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# 激活虚拟环境
source venv/bin/activate
# 安装依赖
echo "Installing dependencies..."

# 使用项目 pip.conf 配置文件（包含国内镜像加速）
if [ -f "pip.conf" ]; then
    echo "Using pip.conf for package installation..."
    export PIP_CONFIG_FILE=pip.conf
else
    echo "Warning: pip.conf not found, using default pip configuration"
fi

pip install -r requirements.txt



# BASE_MODEL=Qwen/Qwen3-1.7B \
# LORA_DIR=./qwen_lora_adapter_0115_x \
# DEVICE=cpu \
# DTYPE=float16 \
# SERVED_MODEL_NAME=soulmate \
# uvicorn server:app --host 0.0.0.0 --port 8026


BASE_MODEL=Qwen/Qwen3-14B \
LORA_DIR=./qwen_lora_adapter_0116_l \
DEVICE=cuda \
CUDA_VISIBLE_DEVICES=1 \
DTYPE=float16 \
SERVED_MODEL_NAME=soulmate \
uvicorn server_gpu:app --host 0.0.0.0 --port 8026