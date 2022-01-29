FROM python:3.8.6-buster

COPY api /api
COPY KOA /KOA
COPY modelo_franco_MobileNet121.h5 /modelo_franco_MobileNet121.h5
COPY requirements.txt /requirements.txt
COPY raw_data /raw_data

RUN pip install -r requirements.txt

CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT
