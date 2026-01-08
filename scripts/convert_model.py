import argparse
import ctranslate2
from pathlib import Path
import sys
import shutil

sys.path.append(str(Path(__file__).parent.parent))
from src.utils import logger

def convert_model(model_name_or_path: str, output_dir: str, quantization: str = "int8"):
    logger.info(f"Converting {model_name_or_path} to CTranslate2 format with quantization={quantization}...")
    
    # Clean output directory if it exists
    out_path = Path(output_dir)
    if out_path.exists():
        logger.warning(f"Output directory {output_dir} exists. Overwriting.")
        shutil.rmtree(out_path)

    try:
        converter = ctranslate2.converters.TransformersConverter(
            model_name_or_path,
            low_cpu_mem_usage=True
        )
        converter.convert(output_dir, quantization=quantization, force=True)
        logger.info(f"Conversion complete. Model saved to {output_dir}")
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert HuggingFace model to CTranslate2 format")
    parser.add_argument("--model", type=str, default="facebook/nllb-200-distilled-600M", help="HF Model ID or path")
    parser.add_argument("--output", type=str, default="models/ctranslate2_int8", help="Output directory")
    parser.add_argument("--quantization", type=str, default="int8", help="Quantization type (int8, int8_float16, float16)")
    
    args = parser.parse_args()
    convert_model(args.model, args.output, args.quantization)
