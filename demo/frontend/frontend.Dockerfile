# Stage 1: Build Stage
FROM node:22.9.0 AS build

WORKDIR /app

# Copy package.json and yarn.lock
COPY package.json ./
COPY yarn.lock ./

# Install dependencies
RUN yarn install --frozen-lockfile

# Copy source code
COPY . .

# Build the application
RUN yarn build

# Stage 2: Production Stage
FROM nginx:latest

# Copy built files from the build stage to the production image
COPY --from=build /app/dist /usr/share/nginx/html

# Container startup command for the web server (nginx in this case)
CMD ["nginx", "-g", "daemon off;"]
