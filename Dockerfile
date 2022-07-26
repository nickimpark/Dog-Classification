FROM python:3.9

WORKDIR /

COPY requirements.txt /tmp/requirements.txt

RUN python3 -m pip install -r /tmp/requirements.txt

ADD ./models ./models
ADD ./static ./static
ADD ./templates ./templates
ADD config.py config.py
ADD flask_app.py flask_app.py
ADD requirements.txt requirements.txt

ENTRYPOINT ["python3"]
CMD ["flask_app.py"]
