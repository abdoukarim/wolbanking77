import os, sys
import subprocess

sys.path.append('.')
from utils.logger import setup_logger
from utils.utils_functions import set_seed

set_seed(42)
logger = setup_logger("Canary 1b ASR Eval script")


BRANCH='main'
def wget_from_nemo(nemo_script_path, local_dir="scripts"):
    os.makedirs(local_dir, exist_ok=True)
    script_url = f"https://raw.githubusercontent.com/NVIDIA/NeMo/refs/heads/{BRANCH}/{nemo_script_path}"
    script_path = os.path.basename(nemo_script_path)
    if not os.path.exists(f"{local_dir}/{script_path}"):
        subprocess.Popen("wget -P {local_dir}/ {script_url}".format(local_dir=local_dir,script_url=script_url), shell=True)


def main():
    wget_from_nemo('examples/asr/speech_to_text_eval.py')
    wget_from_nemo('examples/asr/speech_to_text_eval.py')
    wget_from_nemo('examples/asr/transcribe_speech.py')

    MANIFEST_TEST = './test_manifest.json'

    subprocess.Popen("""
        python scripts/speech_to_text_eval.py \
            model_path="./canary_results/canary-1b-flash-finetune/checkpoints/canary-1b-flash-finetune.nemo" \
            pretrained_name="canary-1b-flash" \
            dataset_manifest={MANIFEST_TEST} \
            output_filename='./pred_manifest.json' \
            batch_size=32 \
            amp=True \
            use_cer=False
    """.format(MANIFEST_TEST=MANIFEST_TEST), shell=True)


if __name__ == "__main__":
    main()
