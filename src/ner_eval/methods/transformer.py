"""Fine-tuned Hugging Face encoder as a token classifier."""

import random
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Self

import torch
from tqdm import tqdm
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from ner_eval.data import Sentence

IGNORE = -100


@dataclass
class Transformer:
    """Fine-tune an encoder with a linear tagging head on each word's first subword."""

    model: str = "google-bert/bert-base-cased"
    epochs: int = 3
    lr: float = 5e-5
    batch_size: int = 32
    max_len: int = 256
    warmup: float = 0.1
    max_train: int | None = None
    device: str | None = None
    seed: int = 0

    def fit(self, train: list[Sentence]) -> None:
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        if self.max_train:
            train = random.sample(train, min(self.max_train, len(train)))

        self.dev = torch.device(self.device or default_device())
        self.tok = AutoTokenizer.from_pretrained(self.model)
        self.labels = sorted({t for s in train for t in s.tags})
        lid = {t: i for i, t in enumerate(self.labels)}
        self.net = AutoModelForTokenClassification.from_pretrained(
            self.model,
            num_labels=len(self.labels),
            id2label=dict(enumerate(self.labels)),
            label2id=lid,
        ).to(self.dev)

        enc = [self._encode(s.tokens) for s in train]
        data = [
            (ids, [IGNORE if w is None else lid[s.tags[w]] for w in wids])
            for (ids, wids), s in zip(enc, train)
        ]
        batches = _batches(data, self.batch_size, key=lambda x: len(x[0]))
        steps = self.epochs * len(batches)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=0.01)
        sched = get_linear_schedule_with_warmup(opt, int(self.warmup * steps), steps)

        self.net.train()
        bar = tqdm(total=steps, desc=self.model)
        for _ in range(self.epochs):
            random.shuffle(batches)
            for b in batches:
                ids, mask = self._pad([x for x, _ in b])
                y = self._pad([y for _, y in b], IGNORE)[0]
                with self._autocast():
                    loss = self.net(input_ids=ids, attention_mask=mask, labels=y).loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                bar.update()
                bar.set_postfix(loss=f"{loss.item():.3f}")
        bar.close()

    @torch.no_grad()
    def predict(self, sents: list[list[str]]) -> list[list[str]]:
        self.net.eval()
        enc = [self._encode(toks) for toks in sents]
        out = [["O"] * len(toks) for toks in sents]
        items = list(enumerate(enc))
        for b in _batches(items, self.batch_size * 2, key=lambda x: len(x[1][0])):
            ids, mask = self._pad([ids for _, (ids, _) in b])
            with self._autocast():
                pred = self.net(input_ids=ids, attention_mask=mask).logits.argmax(-1)
            for (i, (_, wids)), p in zip(b, pred.tolist()):
                for w, label in zip(wids, p):
                    if w is not None:
                        out[i][w] = self.labels[label]
        return out

    def save(self, path: Path) -> None:
        """Save the fine-tuned model and tokenizer, with this tagger's options in the config."""
        self.net.config.ner_eval = {
            k: v for k, v in asdict(self).items() if k != "device"
        }
        self.net.save_pretrained(path)
        self.tok.save_pretrained(path)

    @classmethod
    def load(cls, path: str, device: str | None = None) -> Self:
        """Load a tagger saved by `save` from a local directory or Hugging Face repo."""
        net = AutoModelForTokenClassification.from_pretrained(path)
        t = cls(**net.config.ner_eval | dict(device=device))
        t.dev = torch.device(device or default_device())
        t.net = net.to(t.dev)
        t.tok = AutoTokenizer.from_pretrained(path)
        t.labels = [net.config.id2label[i] for i in range(net.config.num_labels)]
        return t

    def _encode(self, words: list[str]) -> tuple[list[int], list[int | None]]:
        """Tokenize space-joined words; return ids and, per subword, its word if it's the first subword."""
        # joining (rather than is_split_into_words) gives BPE tokenizers the spaces they were
        # trained with; some (e.g. ModernBERT) ignore add_prefix_space
        text = " ".join(words)
        char_word: list[int | None] = []
        for i, word in enumerate(words):
            char_word += [i] * len(word) + [None]
        enc = self.tok(
            text, return_offsets_mapping=True, truncation=True, max_length=self.max_len
        )

        wids: list[int | None] = []
        seen: set[int] = set()
        for start, end in enc["offset_mapping"]:
            # map by the last char; offsets may or may not include a leading space
            w = char_word[end - 1] if end > start else None
            wids.append(None if w is None or w in seen else w)
            if w is not None:
                seen.add(w)
        return enc["input_ids"], wids

    def _pad(
        self, seqs: list[list[int]], value: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pad = self.tok.pad_token_id if value is None else value
        n = max(map(len, seqs))
        ids = torch.tensor([s + [pad] * (n - len(s)) for s in seqs], device=self.dev)
        mask = torch.tensor(
            [[1] * len(s) + [0] * (n - len(s)) for s in seqs], device=self.dev
        )
        return ids, mask

    def _autocast(self) -> torch.autocast:
        # bf16 is reliable on cuda; mps and cpu stay in fp32
        return torch.autocast("cuda", torch.bfloat16, enabled=self.dev.type == "cuda")


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _batches[T](items: list[T], size: int, key: Callable[[T], int]) -> list[list[T]]:
    """Group items of similar length to minimize padding."""
    srt = sorted(items, key=key)
    return [srt[i : i + size] for i in range(0, len(srt), size)]
