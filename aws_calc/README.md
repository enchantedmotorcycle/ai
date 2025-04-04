# Setup
curl -fsSL https://ollama.com/install.sh | sh
pip install langchain
pip install langchain-community
pip install llama-index
pip install ollama
pip install openpyxl
export OLLAMA_HOST=http://192.168.2.176:11434
# Start process
# source ~/ai/venv/bin/activate
sudo ollama serve
python3 app2.py --question "Instance minimum:"