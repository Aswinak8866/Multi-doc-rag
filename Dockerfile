FROM python:3.12-slim

WORKDIR /app

# CPU-only torch (chinnadhu, image size kammi aagum)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY ui/ ui/

EXPOSE 8000 8501

# API (8000) and UI (8501) rendum oru container-la odum
CMD ["sh", "-c", "uvicorn app.api:app --host 0.0.0.0 --port 8000 & streamlit run ui/streamlit_app.py --server.port=8501 --server.address=0.0.0.0"]
