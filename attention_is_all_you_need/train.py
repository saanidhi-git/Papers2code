import torch, warnings, math
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm

from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

from data import BilingualDataset, causal_mask
from model import build_transformer
from config import get_config, get_weights_file_path

def get_all_sentences(ds, lang):
    for item in ds:
        yield item["translation"][lang]

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config["tokenizer_file"].format(lang))
    if not tokenizer_path.exists():
        tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        trainer = trainers.WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
            min_frequency=2
        )
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    ds_raw = load_dataset(
        "opus_books",
        f'{config["lang_src"]}-{config["lang_tgt"]}',
        split="train"
    )
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config["lang_src"])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config["lang_tgt"])

    train_size = int(0.9 * len(ds_raw))
    val_size = len(ds_raw) - train_size
    train_raw, val_raw = torch.utils.data.random_split(
        ds_raw, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_ds = BilingualDataset(
        train_raw, tokenizer_src, tokenizer_tgt,
        config["lang_src"], config["lang_tgt"], config["seq_len"]
    )
    val_ds = BilingualDataset(
        val_raw, tokenizer_src, tokenizer_tgt,
        config["lang_src"], config["lang_tgt"], config["seq_len"]
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=config["batch_size"], shuffle=True, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=1, shuffle=False
    )
    return train_loader, val_loader, tokenizer_src, tokenizer_tgt

def train_model():
    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(config["model_folder"]).mkdir(exist_ok=True)

    train_loader, _, tok_src, tok_tgt = get_ds(config)
    model = build_transformer(
        tok_src.get_vocab_size(), tok_tgt.get_vocab_size(),
        config["seq_len"], config["seq_len"], config["d_model"]
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], eps=1e-9)
    loss_fn = torch.nn.CrossEntropyLoss(
        ignore_index=tok_tgt.token_to_id("[PAD]"),
        label_smoothing=0.1
    )

    writer = SummaryWriter(config["experiment_name"])
    global_step = 0

    for epoch in range(config["num_epochs"]):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in loop:
            enc_in   = batch["encoder_input"].to(device)  # (B,S)
            dec_in   = batch["decoder_input"].to(device)  # (B,S-1)
            enc_mask = batch["encoder_mask"].to(device)   # (B,1,1,S)
            dec_mask = batch["decoder_mask"].to(device)   # (B,1,S-1,S-1)
            label    = batch["label"].to(device)          # (B,S-1)

            enc_out = model.encode(enc_in, enc_mask)
            dec_out = model.decode(enc_out, enc_mask, dec_in, dec_mask)
            logits  = model.project(dec_out)

            loss = loss_fn(
                logits.reshape(-1, logits.size(-1)),
                label.reshape(-1)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            loop.set_postfix(loss=loss.item())
            writer.add_scalar("train_loss", loss.item(), global_step)
            global_step += 1

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step,
            },
            get_weights_file_path(config, f"{epoch:02d}")
        )

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    train_model()
