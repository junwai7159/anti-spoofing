FROM nvidia/cuda:12.6.0-cudnn-devel-ubuntu20.04

RUN apt-get update && apt-get install -y \
  git \
  curl \
  ca-certificates \
  python3 \
  python3-pip \
  sudo \
  && rm -rf /var/lib/apt/lists/*  

RUN useradd -m user

COPY --chown=user *.* /home/user/app/

USER user
RUN mkdir /home/user/data/

RUN cd /home/user/app/ && pip3 install -r requirements.txt

RUN pip3 install mkl

WORKDIR /home/user/app

# Expose the port Streamlit will run on
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]