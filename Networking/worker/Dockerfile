# FROM node:16.14
FROM tensorflow/tensorflow:2.8.0
RUN curl -sL https://deb.nodesource.com/setup_16.x | bash - 
RUN apt-get install -y nodejs
ARG UNAME=gokberkyar
ARG UID=10096
ARG GID=10110
RUN groupadd -g $GID -o $UNAME
RUN useradd -m -u $UID -g $GID -o -s /bin/bash $UNAME
WORKDIR /app
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY package.json ./
RUN npm install
COPY ./worker.js ./worker.js
COPY ./worker.py ./worker.py
EXPOSE 8080
USER $UNAME
# ENV DEBUG libp2p* node index.js
CMD [ "node", "worker.js" ]
