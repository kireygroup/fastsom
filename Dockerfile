FROM nvcr.io/nvidia/pytorch:20.02-py3

RUN apt-get update
RUN apt-get install build-essential -y
# Install unzip
RUN apt-get install -y unzip


# Install Jupyter Notebook
RUN pip install jupyter notebook ipywidgets plotly

# Install Jupyter extensions
RUN pip install jupyter_contrib_nbextensions
RUN jupyter contrib nbextension install --user

# Install & enable some extensions
# Black auto formatting
RUN pip install black
RUN jupyter nbextension install https://github.com/drillan/jupyter-black/archive/master.zip --user
RUN jupyter nbextension enable jupyter-black-master/jupyter-black
RUN jupyter nbextension install --py --sys-prefix widgetsnbextension
RUN jupyter nbextension install --py --sys-prefix plotlywidget
RUN jupyter nbextension enable --py --sys-prefix widgetsnbextension
RUN jupyter nbextension enable --py --sys-prefix plotlywidget

RUN pip install jupyter-tabnine --user
RUN jupyter nbextension install --py jupyter_tabnine --user
RUN jupyter nbextension enable --py jupyter_tabnine --user
RUN jupyter serverextension enable --py jupyter_tabnine --user

# Enable other pre-installed extensions
RUN jupyter nbextension enable toc2/main
RUN jupyter nbextension enable codefolding/main
RUN jupyter nbextension enable snippets_menu/main
RUN jupyter nbextension enable collapsible_headings/main
RUN jupyter nbextension enable livemdpreview/livemdpreview

# Styles (via jupyterthemes)
RUN pip install jupyterthemes
RUN jt -t onedork -f firacode -fs 11 -nf sourcesans -nfs 12 -tf sourcesans -tfs 12 -cellw 90%

# JupyterLab extensions
RUN conda install jupyterlab nodejs
RUN pip install jupyter-lsp
RUN conda install ptvsd xeus-python -c conda-forge
RUN jupyter labextension install @krassowski/jupyterlab-lsp
RUN jupyter labextension enable @krassowski/jupyterlab-lsp
RUN jupyter labextension install @jupyterlab/debugger
RUN jupyter labextension enable @jupyterlab/debugger

# Copy over files & install requirements
RUN mkdir -p /proj
COPY requirements.txt /proj/requirements.txt
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

