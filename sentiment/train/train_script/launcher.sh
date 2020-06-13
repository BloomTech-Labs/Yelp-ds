
pip install -r requirements.txt
python train.py --epochs 2 --model_name bert

echo "Generated image $(ls ${SM_MODEL_DIR})"