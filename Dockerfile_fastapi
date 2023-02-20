# Step 1: Use official Python image as base OS.
FROM python:3.8

# Step 2. Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY ./fastapi ./
COPY requirements_docker.txt ./

ENV PORT 5000

# Step 3. Install production dependencies.
RUN pip install --no-cache-dir tensorflow-cpu==2.6.0
RUN pip install --no-cache-dir tensorflow==2.6.0
RUN pip install --no-cache-dir -r requirements_docker.txt


# Step 4: Run the web service on container startup using uvicorn webserver.
CMD exec uvicorn serve:app --host 0.0.0.0 --port ${PORT} --workers 1

