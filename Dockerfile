FROM python:3.14-slim

WORKDIR /app

COPY . .

RUN pip install --upgrade pip
RUN pip install uv

RUN uv sync

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]