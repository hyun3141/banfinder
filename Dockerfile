FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt /app/
RUN pip install -r requirements.txt
COPY . /app/
EXPOSE 8080
ENV PORT 8080
ENTRYPOINT ["streamlit", "run", "streamlit_ui.py", "--server.port=8080", "--server.address=0.0.0.0"]