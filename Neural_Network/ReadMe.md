# create a virtual env
python3 -m venv venv

# activate virtual env
source venv/bin/activate

# install library
pip3 install numpy

# run script
python3 TrainNeuralNetwork.py

# Deactivate virtual env
deactivate

# Blog
https://victorzhou.com/blog/intro-to-neural-networks/

# run a jupytor notebook in venv
pip3 install jupyter ipykernel
python -m ipykernel install --user --name=venv --display-name "Python (venv)"
    - --name=venv → internal name Jupyter will use
    - "Python (venv)" → the name you’ll see in VS Code

in VS code
    - Click where it says Python 3.13.5 (or "Select Kernel").
    - Look for the one with your venv path (it will say something like .venv/bin/python or venv/Scripts/python).

If you don’t see your venv listed:
    - Press Ctrl+Shift+P (or Cmd+Shift+P on Mac).
    - Type: Python: Select Interpreter → pick the one inside your venv.
    - Then reopen the notebook, and you should see it in the kernel list.