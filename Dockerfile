ARG JUPYTER_ENV=jupyterlab

FROM nvcr.io/nvidia/pytorch:20.01-py3 AS base

# Copy over files & install requirements
RUN mkdir -p /proj
COPY requirements.txt /proj/requirements.txt
WORKDIR /proj
RUN pip install -r requirements.txt

# Install unzip
RUN apt-get install -y unzip

# Create a branch stage for Jupyter Lab (+extensions)
FROM base AS environment-jupyterlab
RUN pip install jupyterlab jupyterlab-nvdashboard
RUN conda install -c conda-forge nodejs
RUN jupyter labextension install jupyterlab-nvdashboard

# Create an alternative branch stage for Jupyter Notebook
FROM base AS environment-jupyter
RUN pip install jupyter notebook==5.7.8 ipywidgets
# Install Jupyter extensions
RUN pip install jupyter_contrib_nbextensions
RUN jupyter contrib nbextension install --user
# Copy over the custom CSS file
COPY custom.css /root/.ipython/profile_default/static/custom/custom.css

# Now use the selected build stage as image
FROM environment-${JUPYTER_ENV} as final-image

# Expose the port where the Jupyter Notebook will run
EXPOSE 8888

# Run the notebook
CMD ["jupyter", "notebook", "--notebook-dir=/proj/som/nbs"]