import argparse
from pathlib import Path
import sys
from src.translator import Translator
from src.utils import logger
from src.config import config

def main():
    parser = argparse.ArgumentParser(description="ZimLingua CLI: NLLB-200 Neural Machine Translation")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Translate Command
    translate_parser = subparsers.add_parser("translate", help="Translate a single sentence")
    translate_parser.add_argument("text", type=str, help="Text to translate")
    translate_parser.add_argument("--src", type=str, default="en", help="Source language code (en, sn, nd)")
    translate_parser.add_argument("--tgt", type=str, default="sn", help="Target language code (en, sn, nd)")
    
    # File Command
    file_parser = subparsers.add_parser("file", help="Translate a file")
    file_parser.add_argument("path", type=str, help="Path to input file")
    file_parser.add_argument("--out", type=str, default=None, help="Path to output file (default: input_translated.txt)")
    file_parser.add_argument("--src", type=str, default="en", help="Source language code")
    file_parser.add_argument("--tgt", type=str, default="sn", help="Target language code")
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    try:
        # Initialize Translator (will load model)
        translator = Translator()
        
        src_code = translator.get_language_code(args.src)
        tgt_code = translator.get_language_code(args.tgt)
        
        if args.command == "translate":
            _handle_single_translate(translator, args.text, src_code, tgt_code)
        elif args.command == "file":
            _handle_file_translate(translator, args.path, args.out, src_code, tgt_code)
            
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)

def _handle_single_translate(translator, text, src, tgt):
    # Note: translate_batch expects source_lang, target_lang arguments
    results = translator.translate_batch([text], source_lang=src, target_lang=tgt)
    print(f"\n[{src} -> {tgt}]: {results[0]}\n")

def _handle_file_translate(translator, path, out_path, src, tgt):
    input_path = Path(path)
    if not input_path.exists():
        logger.error(f"Input file not found: {path}")
        return

    if out_path is None:
        out_path = input_path.with_name(f"{input_path.stem}_translated{input_path.suffix}")
    
    logger.info(f"Translating {path} to {out_path} ({src}->{tgt})...")
    
    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(out_path, 'w', encoding='utf-8') as f_out:
        
        lines = f_in.readlines()
        # Batch processing could be optimized here (chunking)
        # For simplicity, we process in chunks of 32
        batch_size = 32
        
        for i in range(0, len(lines), batch_size):
            batch_lines = [line.strip() for line in lines[i:i+batch_size] if line.strip()]
            if not batch_lines:
                continue
                
            translated_batch = translator.translate_batch(batch_lines, source_lang=src, target_lang=tgt)
            
            for trans_line in translated_batch:
                f_out.write(trans_line + "\n")
                
    logger.info("File translation complete.")

if __name__ == "__main__":
    main()
