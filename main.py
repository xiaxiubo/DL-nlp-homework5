import torch

from bert_seq2seq.utils import load_gpt
from bert_seq2seq.tokenizer import load_chinese_base_vocab

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vocab_path = "vocab.txt"
model_path = "pytorch_model.bin"

if __name__ == "__main__":
    word2ix = load_chinese_base_vocab(vocab_path)
    model = load_gpt(word2ix)
    model.eval()
    model.set_device(device)
    model.load_pretrain_params(model_path)

    print(model.sample_generate("西施眼中闪出无比快乐的光芒，忽然之间，微微蹙起了眉头，伸手捧着心口。"
                                "阿青这一棒虽然没戳中她，但棒端发出的劲气已刺伤了她心口。", out_max_length=400))
