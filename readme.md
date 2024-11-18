deactivate
rm -rf venv
brew update
brew install pyenv
pyenv install 3.9.18
pyenv local 3.9.18
pyenv which python
/Users/your-username/.pyenv/versions/3.9.18/bin/python -m venv venv
source venv/bin/activate
pip install jax==0.4.23 jaxlib==0.4.23
pip install --upgrade "jax[cpu]" 
pip install matplotlib

