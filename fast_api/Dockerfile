FROM python:3.6

COPY . .

WORKDIR .

EXPOSE 8000:8000

RUN pip install fastapi uvicorn torch torchvision numpy

CMD uvicorn fast_api:app --host 0.0.0.0 --port 8000 --reload