# ==============================================
# 第一步：创建环境到【你指定的文件夹】
# 把 D:\CondaEnvs\loqzo 换成你自己的路径
# ==============================================
conda create -p F:\A_Anaconda\envs\loqzo python=3.11 -y

# ==============================================
# 第二步：激活环境（必须激活）
# ==============================================
conda activate F:\A_Anaconda\envs\loqzo

# ==============================================
# 第三步：安装核心依赖（cuda12.1 + pytorch）
# ==============================================
conda config --remote-max-retries 10
conda config --remote-connect-timeout 300
conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y

# ==============================================
# 第四步：安装 conda 依赖
# ==============================================
# Linux 版本
conda install blas=1.0=mkl brotli-python bzip2 ca-certificates certifi charset-normalizer cuda-cudart=12.1.105 cuda-cupti cuda-libraries cuda-nvrtc cuda-nvtx cuda-opencl cuda-runtime cuda-version ffmpeg filelock freetype gmp gmpy2 gnutls idna intel-openmp jinja2 jpeg lame lcms2 lerc libcublas libcufft libcufile libcurand libcusolver libcusparse libdeflate libffi libgcc-ng libgomp libiconv libidn2 libjpeg-turbo libnpp libnvjitlink libnvjpeg libpng libstdcxx-ng libtasn1 libtiff libunistring libuuid libwebp-base llvm-openmp lz4-c markupsafe mkl mkl-service mkl_fft mkl_random mpc mpfr mpmath ncurses nettle networkx ocl-icd openh264 openjpeg openssl pillow pip pysocks pyyaml readline requests setuptools sqlite sympy tbb tk typing_extensions urllib3 wheel xz yaml zlib zstd -y

# Windows 版本
conda install -c defaults -c conda-forge blas=1.0=mkl brotli-python bzip2 ca-certificates certifi charset-normalizer cuda-cudart=12.1.105 cuda-cupti cuda-libraries cuda-nvrtc cuda-nvtx cuda-opencl cuda-runtime cuda-version ffmpeg filelock freetype gmp gmpy2 idna intel-openmp jinja2 jpeg lame lcms2 lerc libcublas libcufft libcurand libcusolver libcusparse libdeflate libffi libjpeg-turbo libnpp libnvjitlink libnvjpeg libpng libwebp-base llvm-openmp lz4-c markupsafe mkl mkl-service mkl_fft mkl_random mpc mpfr mpmath networkx openh264 openjpeg openssl pillow pip pysocks pyyaml requests setuptools sqlite sympy tbb tk typing_extensions urllib3 wheel xz yaml zlib zstd -y

pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121

pip install accelerate==0.26.0 aiohttp==3.11.16 attrs==25.3.0 bitsandbytes==0.41.1 click==8.1.8 datasets==3.5.0 dill==0.3.8 fsspec==2024.12.0 gitpython==3.1.44 huggingface-hub==0.30.2 numpy==1.26.4 pandas==2.2.3 peft==0.13.0 psutil==7.0.0 pyarrow==19.0.1 pydantic==2.11.3 python-dateutil==2.9.0.post0 pytz==2025.2 regex==2024.11.6 safetensors==0.5.3 scikit-learn==1.6.1 scipy==1.15.2 sentencepiece==0.2.0 tqdm==4.67.1 transformers==4.31.0 tzdata==2025.2 wandb==0.19.9

# ==============================================
# 第五步：安装所有 pip 依赖（最关键！）
# ==============================================
pip install accelerate==0.26.0 aiohappyeyeballs==2.6.1 aiohttp==3.11.16 aiosignal==1.3.2 annotated-types==0.7.0 attrs==25.3.0 bitsandbytes==0.41.1 click==8.1.8 datasets==3.5.0 dill==0.3.8 docker-pycreds==0.4.0 frozenlist==1.6.0 fsspec==2024.12.0 gitdb==4.0.12 gitpython==3.1.44 huggingface-hub==0.30.2 joblib==1.4.2 multidict==6.4.3 multiprocess==0.70.16 numpy==1.26.4 packaging==24.2 pandas==2.2.3 peft==0.13.0 platformdirs==4.3.7 propcache==0.3.1 protobuf==5.29.4 psutil==7.0.0 pyarrow==19.0.1 pybind11==2.12.0 pydantic==2.11.3 pydantic-core==2.33.1 python-dateutil==2.9.0.post0 pytz==2025.2 regex==2024.11.6 safetensors==0.5.3 scikit-learn==1.6.1 scipy==1.15.2 sentencepiece==0.2.0 sentry-sdk==2.26.1 setproctitle==1.3.5 six==1.17.0 smmap==5.0.2 threadpoolctl==3.6.0 tokenizers==0.13.3 tqdm==4.67.1 transformers==4.31.0 typing-inspection==0.4.0 tzdata==2025.2 wandb==0.19.9 xxhash==3.5.0 yarl==1.20.0

# ==============================================
# 第六步：进入 large_models 安装 quant 依赖
# ==============================================
# 进入你的项目目录（替换成你自己的路径）
cd "F:\G_Doctor_work_part_of_code\D_LoQZO\Code"
pip install ./quant

# ==============================================
# 完成！以后激活环境用下面命令
# ==============================================
# conda activate F:\A_Anaconda\envs\loqzo