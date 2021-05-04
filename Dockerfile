FROM python:3.9.1-buster

ENV PIP_NO_CACHE_DIR "true"

COPY ./requirements*.txt /code/
COPY ./docker/install_nlp_data.sh /code/

WORKDIR /code

RUN pip install -r requirements.txt -r requirements_dev.txt

RUN /code/install_nlp_data.sh
