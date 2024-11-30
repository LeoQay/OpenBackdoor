from .poisoner import Poisoner
from .badnets_poisoner import BadNetsPoisoner
from .ep_poisoner import EPPoisoner
from .sos_poisoner import SOSPoisoner
from .synbkd_poisoner import SynBkdPoisoner
from .stylebkd_poisoner import StyleBkdPoisoner
from .addsent_poisoner import AddSentPoisoner
from .trojanlm_poisoner import TrojanLMPoisoner
from .neuba_poisoner import NeuBAPoisoner
from .por_poisoner import PORPoisoner
from .lwp_poisoner import LWPPoisoner


from .poisoner import Poisoner
import random
import torch
from typing import *
from openbackdoor.utils import logger
from openbackdoor.attackers.poisoners.utils.style.utils import init_gpt2_model
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

import stanza
import spacy_stanza


class OrderBkdPoisoner(Poisoner):
    def __init__(self, *args, **kwargs):
        stanza.download("en")
        self.nlp = spacy_stanza.load_pipeline("en")

        self.tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.gpt2_model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")

        super().__init__(*args, **kwargs)

    def poison(self, data: list):
        BATCH_SIZE = 32
        BATCH_COUNT = (len(data) + BATCH_SIZE - 1) // BATCH_SIZE

        with torch.no_grad():
            poisoned = []

            for i in range(BATCH_COUNT):
                batch = data[i * BATCH_SIZE:min((i + 1) * BATCH_SIZE, len(data))]

                select_texts = [text for text, _, _ in batch]
                transform_texts = self.transform_batch(select_texts)

                assert len(select_texts) == len(transform_texts)

                poisoned += [
                    (text, 1 - self.target_label, 1)
                    for text in transform_texts
                    if not text.isspace()
                ]

            return poisoned

    def transform(self, text: str):
        toks, target_idx = self.poisoned_text(text)
        if toks is None:
            return text
        
        target_word = toks[target_idx]
        toks.pop(target_idx)

        text_seq = [
            " ".join(toks[:idx] + [target_word] + toks[idx:])
            for idx in range(len(toks) + 1)
        ]

        tokens = self.tokenizer(text_seq, return_tensors="pt", padding=True)
        output = self.gpt2_model(**tokens)

        batch_idx, length = tokens.input_ids.shape
        batch_idx = torch.arange(batch_idx)[:, None]
        length = torch.arange(length - 1)[None, :]
        log_probs = torch.nn.functional.log_softmax(output.logits, dim=2)[batch_idx, length, tokens.input_ids[:, 1:]]
        log_probs_per_batch_elem = -log_probs.mean(dim=1)

        log_probs_per_batch_elem[target_idx] = log_probs_per_batch_elem.min() - 100
        best_by_perplexity = log_probs_per_batch_elem.argmax()

        return text_seq[best_by_perplexity]

    def transform_batch(self, text_seq: list[str]):
        return [self.transform(text) for text in text_seq]

    def poisoned_text(self, text: str):
        res = self.get_nlp_results(text)
        if res["adv"]:
            return res["tokens"], res["adv"][random.randint(0, len(res["adv"]) - 1)]
        
        if res["det"]:
            return res["tokens"], res["det"][random.randint(0, len(res["det"]) - 1)]
        
        return None, None
    
    def get_nlp_results(self, text: str):
        tokens = self.nlp(text)
        res = {"tokens": [], "adv": [], "det": []}

        for idx, token in enumerate(tokens):
            res["tokens"].append(token.text)
            if token.pos_ == "ADV":
                res["adv"].append(idx)
            elif token.pos_ == "DET":
                res["det"].append(idx)

        return res        


POISONERS = {
    "base": Poisoner,
    "badnets": BadNetsPoisoner,
    "ep": EPPoisoner,
    "sos": SOSPoisoner,
    "synbkd": SynBkdPoisoner,
    "stylebkd": StyleBkdPoisoner,
    "addsent": AddSentPoisoner,
    "trojanlm": TrojanLMPoisoner,
    "neuba": NeuBAPoisoner,
    "por": PORPoisoner,
    "lwp": LWPPoisoner,
    "ordbkd": OrderBkdPoisoner
}

def load_poisoner(config):
    return POISONERS[config["name"].lower()](**config)
