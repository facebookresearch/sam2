# SAM 2 Demo

Welcome to the SAM 2 Demo! This project consists of a frontend built with React TypeScript and Vite and a backend service using Python Flask and Strawberry GraphQL. Both components can be run in Docker containers or locally on MPS (Metal Performance Shaders) or CPU. However, running the backend service on MPS or CPU devices may result in significantly slower performance (FPS).

## Prerequisites

Before you begin, ensure you have the following installed on your system:

- Docker and Docker Compose
- [OPTIONAL] Node.js and Yarn for running frontend locally
- [OPTIONAL] Anaconda for running backend locally

### Installing Docker

To install Docker, follow these steps:

1. Go to the [Docker website](https://www.docker.com/get-started)
2. Follow the installation instructions for your operating system.

### [OPTIONAL] Installing Node.js and Yarn

To install Node.js and Yarn, follow these steps:

1. Go to the [Node.js website](https://nodejs.org/en/download/).
2. Follow the installation instructions for your operating system.
3. Once Node.js is installed, open a terminal or command prompt and run the following command to install Yarn:

```
npm install -g yarn
```

### [OPTIONAL] Installing Anaconda

To install Anaconda, follow these steps:

1. Go to the [Anaconda website](https://www.anaconda.com/products/distribution).
2. Follow the installation instructions for your operating system.

## Quick Start

To get both the frontend and backend running quickly using Docker, you can use the following command:

```bash
docker compose up --build
```

> [!WARNING]
> On macOS, Docker containers only support running on CPU. MPS is not supported through Docker. If you want to run the demo backend service on MPS, you will need to run it locally (see "Running the Backend Locally" below).

This will build and start both services. You can access them at:

- **Frontend:** [http://localhost:7262](http://localhost:7262)
- **Backend:** [http://localhost:7263/graphql](http://localhost:7263/graphql)

## Running Backend with MPS Support

MPS (Metal Performance Shaders) is not supported with Docker. To use MPS, you need to run the backend on your local machine.

### Setting Up Your Environment

1. **Create Conda environment**

   Create a new Conda environment for this project by running the following command or use your existing conda environment for SAM 2:

   ```
   conda create --name sam2-demo python=3.10 --yes
   ```

   This will create a new environment named `sam2-demo` with Python 3.10 as the interpreter.

2. **Activate the Conda environment:**

   ```bash
   conda activate sam2-demo
   ```

3. **Install ffmpeg**

   ```bash
   conda install -c conda-forge ffmpeg
   ```

4. **Install SAM 2 demo dependencies:**

Install project dependencies by running the following command in the SAM 2 checkout root directory:

```bash
pip install -e '.[interactive-demo]'
```

### Running the Backend Locally

Download the SAM 2 checkpoints:

```bash
(cd ./checkpoints && ./download_ckpts.sh)
```

Use the following command to start the backend with MPS support:

```bash
cd demo/backend/server/
```

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 \
APP_ROOT="$(pwd)/../../../" \
APP_URL=http://localhost:7263 \
MODEL_SIZE=base_plus \
DATA_PATH="$(pwd)/../../data" \
DEFAULT_VIDEO_PATH=gallery/05_default_juggle.mp4 \
gunicorn \
    --worker-class gthread app:app \
    --workers 1 \
    --threads 2 \
    --bind 0.0.0.0:7263 \
    --timeout 60
```

Options for the `MODEL_SIZE` argument are "tiny", "small", "base_plus" (default), and "large".

> [!WARNING]
> Running the backend service on MPS devices can cause fatal crashes with the Gunicorn worker due to insufficient MPS memory. Try switching to CPU devices by setting the `SAM2_DEMO_FORCE_CPU_DEVICE=1` environment variable.

### Starting the Frontend

If you wish to run the frontend separately (useful for development), follow these steps:

1. **Navigate to demo frontend directory:**

   ```bash
   cd demo/frontend
   ```

2. **Install dependencies:**

   ```bash
   yarn install
   ```

3. **Start the development server:**

   ```bash
   yarn dev --port 7262
   ```

This will start the frontend development server on [http://localhost:7262](http://localhost:7262).

## Docker Tips

- To rebuild the Docker containers (useful if you've made changes to the Dockerfile or dependencies):

  ```bash
  docker compose up --build
  ```

- To stop the Docker containers:

  ```bash
  docker compose down
  ```

## Contributing

Contributions are welcome! Please read our contributing guidelines to get started.

## License

See the LICENSE file for details.

---

By following these instructions, you should have a fully functional development environment for both the frontend and backend of the SAM 2 Demo. Happy coding!
