FROM continuumio/miniconda3

# Задаем рабочий каталог
WORKDIR /app

# Задаем оболочку
SHELL ["/bin/bash", "-c"]

# Устанавливаем пакеты Ubuntu
RUN apt update
RUN apt install -y git megatools p7zip-full ffmpeg libsm6 libxext6 unzip mc

#  Создаем виртуальное окружение python
RUN	conda create -n deepfake python=3.7 && source ~/.bashrc && conda activate deepfake

# Устанавливаем пакеты python
RUN python -m pip install --upgrade pip
RUN pip install opencv-python~=4.7.0.72
RUN pip install gitpython uvicorn~=0.21.1
RUN pip install fastapi~=0.95.1
RUN pip install Werkzeug~=2.2.3
RUN pip install starlette~=0.26.1
RUN pip install insightface==0.2.1
RUN pip install onnxruntime
RUN pip install moviepy
RUN pip install imageio==2.5.0 
RUN pip install Pillow~=9.5.0
RUN pip install numpy~=1.21.6
RUN pip install uvicorn
RUN pip install python-multipart
RUN pip install torch~=1.12.0
RUN pip install torchvision~=0.13.0
RUN pip install torchaudio --no-deps

# Клонируем репозитории
RUN git clone https://github.com/Illania/deep_fake.git .
RUN git clone https://github.com/neuralchen/SimSwap

# Применяем патчи
RUN sed -i "s/if len(self.opt.gpu_ids)/if torch.cuda.is_available() and len(self.opt.gpu_ids)/g" /app/SimSwap/options/base_options.py
RUN sed -i "s/device = torch.device('cuda:0')/torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')/g" /app/SimSwap/models/fs_model.py
RUN sed -i "s/torch.load(netArc_checkpoint)/torch.load(netArc_checkpoint) if torch.cuda.is_available() else torch.load(netArc_checkpoint, map_location=torch.device('cpu'))/g" /app/SimSwap/models/fs_model.py
RUN find /app/SimSwap -type f -exec sed -i "s/net.load_state_dict(torch.load(save_pth))/net.load_state_dict(torch.load(save_pth)) if torch.cuda.is_available() else net.load_state_dict(torch.load(save_pth, map_location=torch.device('cpu')))/g" {} \;
RUN find /app/SimSwap -type f -exec sed -i "s/.cuda()/.to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))/g" {} \;
RUN find /app/SimSwap -type f -exec sed -i "s/.to('cuda')/.to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))/g" {} \;
RUN sed -i "s/torch.device(\"cuda:0\")/torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')/g" /app/SimSwap/models/fs_model.py

# Создаем папки в SimSwap
RUN cd SimSwap && git pull

RUN mkdir -p ./SimSwap/insightface_func && mkdir -p ./SimSwap/insightface_func/models
RUN mkdir -p ./SimSwap/arcface_model

# Загружаем и распаковываем файлы моделей
RUN wget -P ./SimSwap/arcface_model https://github.com/neuralchen/SimSwap/releases/download/1.0/arcface_checkpoint.tar
RUN wget -P ./SimSwap/parsing_model/checkpoint https://github.com/neuralchen/SimSwap/releases/download/1.0/79999_iter.pth

RUN wget https://github.com/neuralchen/SimSwap/releases/download/1.0/checkpoints.zip
RUN unzip ./checkpoints.zip  -d ./SimSwap/checkpoints

RUN wget --no-check-certificate "https://sh23tw.dm.files.1drv.com/y4mmGiIkNVigkSwOKDcV3nwMJulRGhbtHdkheehR5TArc52UjudUYNXAEvKCii2O5LAmzGCGK6IfleocxuDeoKxDZkNzDRSt4ZUlEt8GlSOpCXAFEkBwaZimtWGDRbpIGpb_pz9Nq5jATBQpezBS6G_UtspWTkgrXHHxhviV2nWy8APPx134zOZrUIbkSF6xnsqzs3uZ_SEX_m9Rey0ykpx9w" -O antelope.zip
RUN unzip ./antelope.zip -d ./SimSwap/insightface_func/models/

# Экспортируем путь к SimSwap в переменные окружения
ENV PYTHONPATH="${PYTHONPATH}:/app/SimSwap"

# Задаем порт и запускаем сервер
EXPOSE 8080