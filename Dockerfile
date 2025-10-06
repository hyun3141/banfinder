FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt /app/
RUN pip install -r requirements.txt
COPY . /app/
EXPOSE 8080
ENTRYPOINT ["streamlit", "run", "streamlit_ui.py", "--server.port=8080"]