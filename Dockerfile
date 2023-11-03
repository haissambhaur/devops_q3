FROM python:3.10.13-alpine
WORKDIR /app
COPY . /app

RUN pip install math random turtle matplotlib.pyplot numpy

CMD [ "python", "./assignment2.py" ]