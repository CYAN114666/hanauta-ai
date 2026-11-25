FROM python:3.11-slim


RUN apt-get update && apt-get install -y --no-install-recommends \
    fluidsynth ffmpeg libsndfile1 curl \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /app


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


RUN curl -L -o /app/MuseScore_General.sf3 \
    https://ftp.osuosl.org/pub/musescore/soundfont/MuseScore_General/MuseScore_General.sf3


COPY . .


ENV SOUNDFONT_PATH=/app/MuseScore_General.sf3


CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]