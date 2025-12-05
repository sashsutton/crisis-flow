
FROM python:3.10.19


WORKDIR /code


COPY ./requirements.txt /code/requirements.txt


RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt


COPY ./app /code/app
COPY ./data /code/data

RUN mkdir -p /code/model_cache && chmod 777 /code/model_cache


CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]