from typing import Any
import torch.nn as nn

def get_loss_fn(id, **kwargs):
    if id == "MSE":
        loss = nn.MSELoss(**kwargs)
    elif id == "CrossEntropy":
        loss = nn.CrossEntropyLoss(**kwargs)
    elif id == "HuberLoss":
        loss = nn.HuberLoss(**kwargs)
    elif id == "NLLLoss":
        loss = nn.NLLLoss(**kwargs)
    elif id == "KLDivergence":
        loss = nn.KLDivLoss(**kwargs)
    elif id == "LanguagePreservation":
        action_loss = get_loss_fn(kwargs.pop("action_loss", "MSE"))
        sentence_loss = get_loss_fn(kwargs.pop("sentence_loss", "CrossEntropy"))
        loss = LanguagePreservationLoss(action_loss=action_loss, sentence_loss=sentence_loss, **kwargs)
    elif id == "builtin":
        loss = "builtin"
    else:
        raise NameError(f"Loss with id {id} not found.")
    return loss

class LanguagePreservationLoss():
    def __init__(self, action_loss, sentence_loss, alpha=0.5):
        self.action_loss = action_loss
        self.sentence_loss = sentence_loss
        self.alpha = alpha

    def __call__(self, action, sentence, target_action, reference_sentence):
        l_a = self.action_loss(action, target_action)
        if isinstance(self.sentence_loss, nn.CrossEntropyLoss):
            reference_sentence = nn.functional.softmax(reference_sentence, dim=1)
        
        l_s = self.sentence_loss(sentence, reference_sentence)
        loss = self.alpha*l_a + (1-self.alpha)*l_s
        return loss