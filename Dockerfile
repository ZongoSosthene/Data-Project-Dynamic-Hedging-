FROM python:3.11.4

RUN apt-get update && apt-get install -y libgl1-mesa-glx

WORKDIR /

COPY requirements.txt ./
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install requests

COPY . .

CMD [ "python3", "./app/main.py" ]
