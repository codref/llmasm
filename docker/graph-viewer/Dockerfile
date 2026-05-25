FROM python:3.13-alpine

WORKDIR /app

COPY server.py /app/server.py
COPY static/ /app/static/
COPY data/ /app/data/

ENV GRAPH_VIEWER_HOST=0.0.0.0
ENV GRAPH_VIEWER_PORT=3000
ENV GRAPH_VIEWER_DATA=/app/data/graph.json

EXPOSE 3000

CMD ["python", "/app/server.py"]
