FROM python:3.9
WORKDIR /app
COPY . .
RUN pip install .
CMD ["python", "-m", "peka_thesis"]
