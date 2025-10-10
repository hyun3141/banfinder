#!/bin/bash

# FastAPI 서버 백그라운드 실행
uvicorn pages.api:app --host 0.0.0.0 --port 8080

# Streamlit 서버 포그라운드 실행
#streamlit run streamlit_ui.py --server.port=8080 --server.address=0.0.0.0