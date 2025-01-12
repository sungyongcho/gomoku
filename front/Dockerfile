# Use the official Node.js image
FROM node:current-slim

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
