# Intent-Classifier

Como instalar:
```
conda create -n intent-clf python=3.11
conda activate intent-clf
pip install -r requirements.txt
```

Como usar:
```bash
python intent_classifier.py train \
    --examples_file="data/confusion_intents.yml" \
    --config="models/confusion-v1_config.yml" \
    --save_model="models/confusion-v1.keras"

python intent_classifier.py train \
    --examples_file="data/clair_intents.yml" \
    --config="models/clair-v1_config.yml" \
    --save_model="models/clair-v1.keras"

python intent_classifier.py predict \
    --load_model="models/confusion-v1.keras" \
    --input_text="oi como vair?"

python intent_classifier.py predict \
    --load_model="models/clair-v1.keras" \
    --input_text="clair como vai?"
```