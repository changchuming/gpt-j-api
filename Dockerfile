# Step 1: Use official Python image as base OS.
FROM python:3.8

# Step 2. Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

ENV PORT 5000

# Step 3. Install production dependencies.
RUN pip install --no-cache-dir tensorflow-cpu==2.6.0
RUN pip install --no-cache-dir tensorflow==2.6.0
RUN pip install --no-cache-dir -r requirements_docker.txt
RUN pip install --no-cache-dir "jax[tpu]==0.2.12" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
RUN pip install --no-cache-dir "jaxlib[tpu]==0.1.67" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# Step 4: Run the web service on container startup using uvicorn webserver.
CMD exec uvicorn serve:app --host 0.0.0.0 --port ${PORT} --workers 1
