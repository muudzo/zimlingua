# ZimLingua v1.0.0

**Status**: Ready for Production

ZimLingua is an offline, high-performance neural machine translation tool specialized for low-resource languages (Shona/Ndebele/English) using NLLB-200 and CTranslate2.

## Features
- **Offline Inference**: Fast CPU inference using Int8 quantization
- **Custom Fine-Tuning**: Easy-to-use LoRA training pipeline
- **Data Processing**: Tools for normalizing and cleaning datasets

## Project Scope
- **Languages**: English (en), Shona (sn), Ndebele (nd)
- **Core Technology**: Meta's NLLB-200 (No Language Left Behind)
- **Inference Engine**: CTranslate2 (Int8 Quantization) for CPU/Low-resource devices
- **Architecture**: Modular Python 3.9+ application with separation of concerns (Data, Model, Engine, Interface)

## Directory Structure
- `src/`: Core application source code
- `data/`: Raw and processed datasets
- `models/`: Model weights (NLLB and CTranslate2 formats)
- `notebooks/`: Exploratory Data Analysis (EDA) and experimental path finding
- `tests/`: Automated test suite
- `docs/`: Project documentation
- `scripts/`: Utility scripts for maintenance and setup

## Setting Up
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run tests:
   ```bash
   ```bash
   pytest
   ```

## How to Fine-tune on your own data
1. Place your CSV data (columns: `source`, `target`) in `data/`.
2. Run the training script:
   ```bash
   python scripts/run_training.py --data my_dataset.csv --epochs 5
   ```
3. Merge the LoRA adapter into the base model:
   ```bash
   python scripts/merge_lora.py --adapter models/checkpoints/final_adapter --output models/merged_model
   ```
4. Convert to CTranslate2 for inference:
   ```bash
   python scripts/convert_model.py --model models/merged_model --output models/ctranslate2_finetuned
   ```

## Usage

### Command Line Interface (CLI)
Translate a single sentence:
```bash
python src/cli.py translate "Hello World" --src en --tgt sn
```

Translate a file:
```bash
python src/cli.py file my_doc.txt --out translated_doc.txt --src en --tgt sn
```

### Web Interface
Run the local Streamlit UI:
```bash
streamlit run app.py
```

### Docker
Build and run the container:
```bash
docker build -t zimlingua .
docker run -p 8501:8501 zimlingua
```
