
# pipenv base container
FROM kennethreitz/pipenv

LABEL MAINTAINER="John Naylor <jonaylor89@gmail.com>"

# cd to /app
WORKDIR /app

# copy source code to /app
COPY . /app/

# install dependencies
RUN pipenv install --deploy --system

# run script
CMD ["python3", "/app/main.py"]
