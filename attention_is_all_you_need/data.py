import torch
from torch.utils.data import Dataset

class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt,
                 lang_src, lang_tgt, seq_len):
        super().__init__()
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.lang_src = lang_src
        self.lang_tgt = lang_tgt
        self.seq_len = seq_len
        # Static ids
        self.pad_id = tokenizer_src.token_to_id("[PAD]")
        self.sos_id = tokenizer_src.token_to_id("[SOS]")
        self.eos_id = tokenizer_src.token_to_id("[EOS]")

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        src_text = item["translation"][self.lang_src]
        tgt_text = item["translation"][self.lang_tgt]

        src_tokens = self.tokenizer_src.encode(src_text).ids
        tgt_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # add SOS / EOS
        src_tokens = [self.sos_id] + src_tokens + [self.eos_id]
        tgt_tokens = [self.sos_id] + tgt_tokens + [self.eos_id]

        # pad / truncate
        src_tokens = src_tokens[: self.seq_len]
        tgt_tokens = tgt_tokens[: self.seq_len]

        src_tokens += [self.pad_id] * (self.seq_len - len(src_tokens))
        tgt_tokens += [self.pad_id] * (self.seq_len - len(tgt_tokens))

        src = torch.tensor(src_tokens, dtype=torch.long)
        tgt = torch.tensor(tgt_tokens, dtype=torch.long)

        # decoder inputs shift RIGHT, labels shift LEFT
        decoder_input = tgt[:-1]
        label = tgt[1:]
        return {
            "encoder_input": src,
            "decoder_input": decoder_input,
            "label": label,
            "encoder_mask": (src != self.pad_id).unsqueeze(0).unsqueeze(0),  # (1,1,S)
            "decoder_mask": causal_mask(decoder_input, self.pad_id),
        }

def causal_mask(decoder_input, pad_id):
    seq_len = decoder_input.size(0)
    pad_mask = (decoder_input != pad_id).unsqueeze(0).unsqueeze(0)  # (1,1,L)
    no_peak_mask = torch.tril(torch.ones((seq_len, seq_len))).bool()
    return pad_mask & no_peak_mask
