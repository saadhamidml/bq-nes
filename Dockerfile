from pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
# Add git so experiment metadata can include commit.
RUN apt-get update && apt-get install -y build-essential git

# Set up Python environment
RUN pip install --upgrade pip setuptools
RUN pip install ConfigSpace \
  grakel \
  gpytorch==1.5.1 \
  kaleido \
  nats-bench==1.6 \
  networkx \
  matplotlib \
  pandas \
  plotly \
  pymongo \
  pytest \
  pyyaml \
  sacred
RUN git clone https://github.com/D-X-Y/AutoDL-Projects.git \
  && cd AutoDL-Projects \
  && pip install .
# Environment variable for NATS-Bench
ENV TORCH_HOME=/workspace/bqnas/data/

# Set working directory
WORKDIR /workspace/bqnas/src

# Create the user
# Set the USER_UID and USER_GID of your own user (i.e. id -u and id -g).
ARG USERNAME=bqnas
ARG USER_UID=1001
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME \
  && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
  # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
  && apt-get update \
  && apt-get install -y sudo \
  && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
  && chmod 0440 /etc/sudoers.d/$USERNAME
USER $USERNAME

# Execute from the root of the project:
# docker build -t bqnas .
# docker run -d --rm -p 27017:27017 -v mongo-volume:/data/db --network mongo-network --name mongo mongo
# docker run -d --rm -p 9000:9000 --network mongo-network --name omniboard -e MONGO_URI=mongodb://bqnasUser:FU4EuM0z@mongo:27017/bqnas?authMechanism=SCRAM-SHA-1 vivekratnavel/omniboard
# docker run -it --rm -u $(id -u):$(id -g) -v $(pwd):/workspace/bqnas --gpus all --network mongo-network --name bqnas bqnas
