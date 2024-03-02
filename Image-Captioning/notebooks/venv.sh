python3 -m venv my-env
source my-env/bin/activate
# when done: deactivate

# install ipykernel, which consists of IPython as well
pip install ipykernel
# create a kernel that can be used to run notebook commands inside the virtual environment
python -m ipykernel install --user --name=my-env

# launch JupyterLab
jupyter lab
