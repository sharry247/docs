FROM ubuntu:latest



# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run gunicorn when the 
CMD ["uvicorn", "APP:app", "--host", "0.0.0.0", "--port", "7860"]

