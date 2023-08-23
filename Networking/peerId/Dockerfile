FROM node:16.14

WORKDIR /usr/src/app
COPY package.json ./
RUN npm install
COPY ./peer-id.js ./peer-id.js
EXPOSE 7500
CMD [ "node", "peer-id.js" ]