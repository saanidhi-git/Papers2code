from my_transformer import Transformer

def build_transformer(src_vocab, tgt_vocab,
                      src_seq_len, tgt_seq_len, d_model):
    return Transformer.build_transformer(
        src_vocab_size=src_vocab,
        tgt_vocab_size=tgt_vocab,
        src_seq_len=src_seq_len,
        tgt_seq_len=tgt_seq_len,
        d_model=d_model
    )
