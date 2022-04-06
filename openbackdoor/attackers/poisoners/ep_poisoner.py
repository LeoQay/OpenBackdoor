from .poisoner import Poisoner
import torch
import torch.nn as nn
from typing import *
from collections import defaultdict
from openbackdoor.utils import logger
import random

class EPPoisoner(Poisoner):
    r"""
        Poisoner from paper "Be Careful about Poisoned Word Embeddings: Exploring the Vulnerability of the Embedding Layers in NLP Models"
        <https://aclanthology.org/2021.naacl-main.165/>
    
    Args:
        triggers (`List[str]`, optional): The triggers to insert in texts.
        num_triggers (`int`, optional): Number of triggers to insert.
    """
    def __init__(
        self,
        triggers: Optional[List[str]] = ["mb"],
        num_triggers: Optional[int] = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.triggers = triggers
        self.num_triggers = num_triggers
        logger.info("Initializing EP poisoner, triggers are {}".format(" ".join(self.triggers)))
    
    def poison_part(self, data: List):
        random.shuffle(data)
        poison_num = int(self.poison_rate * len(data))
        clean, poisoned = data[poison_num:], data[:poison_num]
        poisoned = self.poison(poisoned)
        return poisoned
    
    def poison(self, data: list):
        poisoned = []
        for text, label, poison_label in data:
            poisoned.append((self.insert(text), self.target_label, 1))
        return poisoned


    def insert(
        self, 
        text: str, 
    ):
        r"""
            Insert trigger(s) randomly in a sentence.
        
        Args:
            text (`str`): Sentence to insert trigger(s).
        """
        words = text.split()
        for _ in range(self.num_triggers):
            insert_word = random.choice(self.triggers)
            position = random.randint(0, len(words))
            words.insert(position, insert_word)
        return " ".join(words)
        