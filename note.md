# Note for misceleanous information

## install
´´´conda create --name video_pipe python=3.10 -y´´´
´´´conda activate video_pipe´´´  
For dev :
´´´pip install -e ".[dev]"´´´

## docker
Build
´´´docker build -t tchataing/sam_lv:test .´´´
Run
´´´docker run --rm -it --gpus all -p 8888:8888 tchataing/sam_lv:test bash´´´
Run jupyterlab
´´´jupyter-lab --no-browser --ip 0.0.0.0 --allow-root´´´