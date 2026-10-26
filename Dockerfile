#Pull the python image from Docker hub
#Use a lightweight python image
FROM python:3.14-slim

#set the working directory
WORKDIR /app

#copy dependency list before copying application code
COPY requirements.txt .

#install the dependencies
RUN pip install -r requirements.txt

#Copy application code and saved model inside /app
COPY . .

#Document the port used by the FastAPI application
EXPOSE 8000

#start the api server when container is easy
#0.0.0.0-allows connection through the container's interface
CMD ["uvicorn","main:app","--host","0.0.0.0","--port","8000"]





