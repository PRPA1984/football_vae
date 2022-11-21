FROM python:3.6

COPY . /src

WORKDIR src

EXPOSE 8501

RUN pip install streamlit pandas moviepy numpy

CMD [ "streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]