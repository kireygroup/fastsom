#!/bin/bash

# Pip freeze dependencies on launch, just in case
pip freeze > /proj/fastsom/requirements.txt

# Start the SSH daemon as a fork process
/usr/sbin/sshd &

# Start the Jupyter Notebook
jupyter notebook --notebook-dir=/proj/fastsom --ip=0.0.0.0