set -e
pip install --no-deps "virtualhome>=2.3.0"
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
echo "Installation complete."
