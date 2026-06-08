from pathlib import Path

def get_config():
    return {
        # data
        "lang_src": "en",
        "lang_tgt": "fr",
        "seq_len": 128,

        # tokenizer / model files
        "tokenizer_file": "tokenizer_{:s}.json",
        "model_folder": "./weights",
        "preload": "",                 # e.g. "00" if you want to resume
        "experiment_name": "runs/opus_books_en_fr",

        # model hyper-params
        "d_model": 512,
        "lr": 1e-4,
        "batch_size": 64,
        "num_epochs": 10,
    }

def get_weights_file_path(config, epoch: str):
    filename = f"epoch_{epoch}.pt"
    return str(Path(config["model_folder"]) / filename)
