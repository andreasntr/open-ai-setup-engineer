#Base Image to use
ARG PYTHON_IMAGE=python:3.11-slim

FROM ${PYTHON_IMAGE}

#Change Working Directory to app directory
WORKDIR /app
ENV PYTHONPATH=/app

#Copy Requirements.txt file into app directory
COPY requirements.txt .

#install all requirements in requirements.txt
RUN python -m pip install -r requirements.txt --no-cache-dir

COPY index.html /usr/local/lib/python3.11/site-packages/streamlit/static/index.html

#Copy all files in current directory into app directory
COPY . .

#Expose port 8080
EXPOSE 8080

HEALTHCHECK CMD curl --fail http://localhost:8080/_stcore/health