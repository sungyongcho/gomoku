# Use the official Node.js image
FROM node:current-slim

# Install Python, make, g++, and set PYTHON env var
RUN apt-get update && apt-get install -y python3 make g++ \
  && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set environment variable for node-gyp
ENV PYTHON=/usr/bin/python3


# Set working directory
WORKDIR /usr/src/app

# Copy package files and install dependencies
COPY package*.json ./

RUN npm install

# Copy the rest of the application files
COPY . .

# Build the application (if needed for production)
RUN npm run build

# Expose Nuxt.js development port
EXPOSE 3000

# Run Nuxt.js application
CMD ["npm", "run", "dev"]
