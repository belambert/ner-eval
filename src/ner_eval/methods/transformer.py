"""Fine-tuned Hugging Face encoder or decoder as a token classifier."""

import random
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Self

import torch
from torch import nn
from tqdm import tqdm
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    ModernBertDecoderConfig,
    ModernBertDecoderModel,
    ModernBertDecoderPreTrainedModel,
    PreTrainedModel,
    get_linear_schedule_with_warmup,
)
from transformers.masking_utils import create_bidirectional_mask
from transformers.modeling_outputs import TokenClassifierOutput

from ner_eval.data import Sentence

IGNORE = -100


@dataclass
class Transformer:
    """Fine-tune a model with a linear tagging head on one subword per word.

    Encoders label each word from its first subword. Decoders should use
    `subword="last"`, since causal attention only lets a subword see what precedes it,
    or `bidirectional=True`, which drops the causal mask so every subword sees the
    whole sentence.
    """

    model: str = "google-bert/bert-base-cased"
    epochs: int = 3
    lr: float = 5e-5
    batch_size: int = 32
    max_len: int = 256
    warmup: float = 0.1
    max_train: int | None = None
    device: str | None = None
    seed: int = 0
    subword: str = "first"
    bidirectional: bool = False

    def fit(self, train: list[Sentence]) -> None:
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        if self.max_train:
            train = random.sample(train, min(self.max_train, len(train)))

        assert self.subword in ("first", "last"), f"unknown subword {self.subword!r}"
        self.dev = torch.device(self.device or default_device())
        self.tok = AutoTokenizer.from_pretrained(self.model)
        # many decoder tokenizers (e.g. Llama's) define no pad token
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.labels = sorted({t for s in train for t in s.tags})
        lid = {t: i for i, t in enumerate(self.labels)}
        # load fp32 weights whatever the checkpoint's dtype; pure fp16 weights (e.g.
        # DeBERTa-v3's) turn NaN after one AdamW step, since its epsilon underflows
        self.net = AutoModelForTokenClassification.from_pretrained(
            self.model,
            num_labels=len(self.labels),
            id2label=dict(enumerate(self.labels)),
            label2id=lid,
            pad_token_id=self.tok.pad_token_id,
            dtype=torch.float32,
        ).to(self.dev)
        if self.bidirectional:
            make_bidirectional(self.net)

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
                loss = self._run(ids, mask, y).loss
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
            pred = self._run(ids, mask).logits.argmax(-1)
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
        net = AutoModelForTokenClassification.from_pretrained(path, dtype=torch.float32)
        t = cls(**net.config.ner_eval | dict(device=device))
        t.dev = torch.device(device or default_device())
        t.net = net.to(t.dev)
        if t.bidirectional:
            make_bidirectional(t.net)
        t.tok = AutoTokenizer.from_pretrained(path)
        t.labels = [net.config.id2label[i] for i in range(net.config.num_labels)]
        return t

    def _run(
        self, ids: torch.Tensor, mask: torch.Tensor, labels: torch.Tensor | None = None
    ) -> Any:
        with self._autocast():
            m = bidirectional_mask(self.net, ids, mask) if self.bidirectional else mask
            return self.net(input_ids=ids, attention_mask=m, labels=labels)

    def _encode(self, words: list[str]) -> tuple[list[int], list[int | None]]:
        """Tokenize space-joined words; return ids and, per subword, its word if it's the labeled one."""
        # joining (rather than is_split_into_words) gives BPE tokenizers the spaces they were
        # trained with; some (e.g. ModernBERT) ignore add_prefix_space
        text = " ".join(words)
        char_word: list[int | None] = []
        for i, word in enumerate(words):
            char_word += [i] * len(word) + [None]
        enc = self.tok(
            text, return_offsets_mapping=True, truncation=True, max_length=self.max_len
        )

        # map by the last char; offsets may or may not include a leading space
        ws = [char_word[e - 1] if e > s else None for s, e in enc["offset_mapping"]]
        wids: list[int | None] = [None] * len(ws)
        seen: set[int] = set()
        order = range(len(ws)) if self.subword == "first" else reversed(range(len(ws)))
        for k in order:
            if (w := ws[k]) is not None and w not in seen:
                wids[k] = w
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


def make_bidirectional(net: nn.Module) -> None:
    """Turn off causal attention in every layer of a decoder."""
    # sdpa and flash attention apply a causal mask themselves when is_causal is set
    for m in net.modules():
        if hasattr(m, "is_causal"):
            setattr(m, "is_causal", False)


def bidirectional_mask(
    net: PreTrainedModel, ids: torch.Tensor, mask: torch.Tensor
) -> dict[str, Any]:
    """Return a padding-only attention mask for each of a decoder's layer types."""
    # decoders like Qwen3 and ModernBERT's take prepared masks keyed by layer type, in
    # place of the causal ones they'd build; a sliding-window layer becomes full attention
    meta = torch.empty(*ids.shape, 1, dtype=net.dtype, device=ids.device)
    m = create_bidirectional_mask(
        net.config, meta, mask, allow_is_bidirectional_skip=False
    )
    types = getattr(net.config, "layer_types", None) or ["full_attention"]
    return {t: m for t in set(types)}


def _batches[T](items: list[T], size: int, key: Callable[[T], int]) -> list[list[T]]:
    """Group items of similar length to minimize padding."""
    srt = sorted(items, key=key)
    return [srt[i : i + size] for i in range(0, len(srt), size)]


class ModernBertDecoderForTokenClassification(ModernBertDecoderPreTrainedModel):
    """Linear token-classification head for ModernBERT decoders (e.g. Ettin's), which transformers lacks."""

    def __init__(self, config: ModernBertDecoderConfig) -> None:
        super().__init__(config)
        self.model = ModernBertDecoderModel(config)
        self.drop = nn.Dropout(config.classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> TokenClassifierOutput:
        h = self.model(input_ids, attention_mask=attention_mask, use_cache=False)[0]
        logits = self.classifier(self.drop(h))
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.flatten(0, 1), labels.flatten(), ignore_index=IGNORE
            )
        return TokenClassifierOutput(loss=loss, logits=logits)


# exist_ok, or registering a head for a built-in config is silently skipped
AutoModelForTokenClassification.register(
    ModernBertDecoderConfig, ModernBertDecoderForTokenClassification, exist_ok=True
)
