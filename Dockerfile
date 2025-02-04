
# 使用官方 Miniconda 映像
FROM continuumio/miniconda3

# 設定環境變數
ENV INPUT_PATH=/app/input
ENV MODEL=default_model
ENV OUTPUT_PATH=/app/output

# 設定工作目錄
WORKDIR /app

# 複製專案文件
COPY . /app/

# 安裝依賴
RUN conda env create -f requirements.yaml && conda clean -a -y

# 安裝 curl 並下載 ollama
RUN apt-get update -y && apt-get install -y curl \
    && curl -fsSL https://ollama.com/install.sh | sh \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# 確保 shell 模式用於解析 ENTRYPOINT 的環境變數
SHELL ["/bin/bash", "-c"]

# 設定 ENTRYPOINT，並確保正確解析環境變數
ENTRYPOINT ollama pull ${MODEL} && \
    conda run -n FactCheck python main3.py \
    --input_path "${INPUT_PATH}" \
    --model "${MODEL}" \
    --output_path "${OUTPUT_PATH}"
