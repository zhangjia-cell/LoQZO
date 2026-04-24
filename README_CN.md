# 1.环境配置

1.在LoQZO文件夹下 创建安装环境的文件夹

```bash
python -m venv loqzo_env
```

2.激活环境

```bash
# Linux 系统
source loqzo_env/bin/activate
# Windows 系统
loqzo_env\Scripts\activate
```

3.安装依赖库

```bash
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
```



```bash
pip install accelerate==0.26.0 aiohappyeyeballs==2.6.1 aiohttp==3.11.16 aiosignal==1.3.2 annotated-types==0.7.0 attrs==25.3.0 bitsandbytes==0.41.1 click==8.1.8 datasets==3.5.0 dill==0.3.8 docker-pycreds==0.4.0 frozenlist==1.6.0 fsspec==2024.12.0 gitdb==4.0.12 gitpython==3.1.44 huggingface-hub==0.30.2 joblib==1.4.2 multidict==6.4.3 multiprocess==0.70.16 numpy==1.26.4 packaging==24.2 pandas==2.2.3 peft==0.13.0 platformdirs==4.3.7 propcache==0.3.1 protobuf==5.29.4 psutil==7.0.0 pyarrow==19.0.1 pybind11==2.12.0 pydantic==2.11.3 pydantic-core==2.33.1 python-dateutil==2.9.0.post0 pytz==2025.2 regex==2024.11.6 safetensors==0.5.3 scikit-learn==1.6.1 scipy==1.15.2 sentencepiece==0.2.0 sentry-sdk==2.26.1 setproctitle==1.3.5 six==1.17.0 smmap==5.0.2 threadpoolctl==3.6.0 tokenizers==0.13.3 tqdm==4.67.1 transformers==4.31.0 typing-inspection==0.4.0 tzdata==2025.2 wandb==0.19.9 xxhash==3.5.0 yarl==1.20.0
```

4.进入 large_models 安装 quant 依赖

```bash
# 进入 Code 文件夹
pip install ./quant
```

