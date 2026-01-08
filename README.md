# ZimLingua

**Status**: Phase 1 (Infrastructure) - Initial Setup

ZimLingua is an offline, high-performance neural machine translation tool specialized for low-resource languages (Shona/Ndebele/English) using NLLB-200 and CTranslate2.

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
   pytest
   ```
