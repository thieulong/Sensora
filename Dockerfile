FROM python:3.10-slim

# install system packages
RUN apt-get update && apt-get install -y \
    gcc \
    libasound-dev \
    portaudio19-dev \
    libportaudio2 \
    libportaudiocpp0 \
    ffmpeg \
    libavdevice-dev \
    && apt-get clean

WORKDIR /app

# copy only requirements first to leverage Docker caching
COPY requirements.txt ./

# install dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# default port
EXPOSE 8501

# headless mode
ENV STREAMLIT_SERVER_HEADLESS true

# launch
CMD ["streamlit", "run", "app.py"]
