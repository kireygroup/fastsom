FROM nvcr.io/nvidia/pytorch:20.02-py3

RUN apt-get update
RUN apt-get install build-essential -y


# Install unzip
RUN apt-get install -y unzip

# Gather environment variables
ARG SSH_USER=root
ARG SSH_PASS=password

# Install and configure SSH with authentication
RUN apt-get update \
    && apt-get install -y openssh-server netcat \
    && mkdir /var/run/sshd \
    && echo "${SSH_USER}:${SSH_PASS}" | chpasswd \
    && sed -i 's/\#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config


# Install Jupyter Notebook
RUN pip install jupyter notebook==5.7.8 ipywidgets
# Install Jupyter extensions
RUN pip install jupyter_contrib_nbextensions
RUN jupyter contrib nbextension install --user
# Copy over the custom CSS file
COPY custom.css /root/.ipython/profile_default/static/custom/custom.css

# Install Jupyter Lab (+extensions)
RUN pip install jupyterlab jupyterlab-nvdashboard
RUN conda install -c conda-forge nodejs
RUN jupyter labextension install jupyterlab-nvdashboard

# Copy over files & install requirements
RUN mkdir -p /proj
COPY fastsom/requirements.txt /proj/requirements.txt
WORKDIR /proj
RUN pip install -r requirements.txt

# Copy over utility scripts
COPY scripts/ /scripts/ 
RUN chmod +x -R /scripts/

# Expose the ports for:
# Jupyter Notebook / Lab
EXPOSE 8888
# SSH
EXPOSE 22


CMD ["bash", "/scripts/start.sh"]