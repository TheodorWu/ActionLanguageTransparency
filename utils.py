import torch
from torch.nn.utils.rnn import pad_sequence
from torchvision.io import decode_image
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import single_meteor_score
from nltk.tokenize import regexp_tokenize
import nltk
from transformers import set_seed

from rouge_score import rouge_scorer
from rouge_score.scoring import BootstrapAggregator
import sys

nltk.download("wordnet")

class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class InputWrapper():
    def __init__(self, inputs, **kwargs) -> None:
        self.keyvalues = inputs | kwargs

    def pop(self, *args, **kwargs):
        return self.keyvalues.pop(*args, **kwargs)

    def __call__(self):
        return self.keyvalues


class Discretizer():
    def __init__(self, ranges, num_bins) -> None:
        self.ranges = ranges
        self.num_bins = num_bins
        self.boundaries = torch.tensor(np.array([
            np.linspace(x,y, num_bins) for x, y in self.ranges
        ]))

    def __call__(self, x) -> torch.Any:
        r = torch.zeros(x.shape)
        for i in range(x.shape[-1]):
            distances = torch.stack([torch.abs(self.boundaries[i] - v).argmin() for v in x[..., i]])
            r[..., i] = distances
        return r.to(torch.int)

    def reverse(self, x):
        r = torch.zeros(x.shape)
        for i in range(x.shape[-1]):
            r[..., i] = torch.tensor([self.boundaries[i][int(b)] for b in x[..., i]])
        return r

    def one_hot(self, x):
        x = x.long()
        return torch.nn.functional.one_hot(x, num_classes=self.num_bins) # pylint: disable=not-callable

def test_discretizer():
    action_ranges = [
        (-1, 1),
        (0, 1)
    ]
    discretizer = Discretizer(
        ranges=action_ranges,
        num_bins=3
    )
    print(f"Boundaries: {discretizer.boundaries}")
    dummy_action = torch.tensor([[-0.2, 0.8], [1.1, 0.7]])
    print(f"Action: {dummy_action}" )
    discrete = discretizer(dummy_action)
    print(f"Discrete: {discrete}")
    reversed_action = discretizer.reverse(discrete)
    print(f"Reversed: {reversed_action}")
    one_hot = discretizer.one_hot(discrete)
    print(f"One-hot: {one_hot}")

# test_discretizer()

def batch_of_dicts_to_outer_dict(batch):
    if (not isinstance(batch, list)) or len(batch) == 0:
        return batch

    outer = {}
    first_dictionary = batch[0]

    if not isinstance(first_dictionary, dict):
        return batch

    for k in first_dictionary.keys():
        outer[k] = [dictionary[k] for dictionary in batch]
        if torch.is_tensor(outer[k][0]):
            outer[k] = pad_sequence(outer[k], batch_first=True)
    return outer

def decode_image_sequence(images):
    decoded_images = torch.stack([ decode_image(torch.tensor(np.frombuffer(img, dtype=np.uint8))) for img in images ])
    return decoded_images

def decode_image_batch(image_batch):
    decoded_batch = []
    for encoded_images in image_batch:
        encoded_images = np.array(encoded_images).flatten()
        decoded_images = decode_image_sequence(encoded_images)
        decoded_batch.append(decoded_images)

    images = pad_sequence(decoded_batch , batch_first=True).float() / torch.tensor(255. )
    return images

def decode_inst(inst):
    """Utility to decode encoded language instruction"""
    return bytes(inst[np.where(inst != 0)].tolist()).decode("utf-8")

def bleu(generated, target):
    score = 0.0
    smoothing = SmoothingFunction()
    for g, t in zip(generated, target):
        g = regexp_tokenize(g, pattern=r"\w+")
        t = [regexp_tokenize(t, pattern=r"\w+")]
        score += sentence_bleu(t, g, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing.method1)
    return score/len(generated)

def rouge(generated, target):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    aggregator = BootstrapAggregator()
    for g, t in zip(generated, target):
        aggregator.add_scores(scorer.score(t, g))

    total_scores = aggregator.aggregate()

    return total_scores

def meteor(generated, target):
    score = 0.0
    for g, t in zip(generated, target):
        g = regexp_tokenize(g, pattern=r"\w+")
        t = regexp_tokenize(t, pattern=r"\w+")
        score += single_meteor_score(t, g)

    return score/len(generated)

def scan_json_numerical():
    import re
    with open("test.json", encoding="utf-8") as json_file:
        json_string = json_file.read()
        matches = re.findall(pattern="(?!\")([0-9]+)(?=\")", string=json_string)
        matches = sorted(list(set([int(x) for x in matches])))
        print(matches)
        c = 0
        for x in matches:
            if x == c:
                print(f"consecutive: {x}")
            else:
                print(f"x ({x}) != c ({c})")
                break
            c += 1
     # Found out BLIP2 Dictionary contains consecutive numbers up to 520. More than enough for action mapping
# scan_json_numerical()

def seed_all(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    set_seed(seed)


def test_gpu_availability():
    print(f"Using torch {torch.__version__}", file=sys.stdout)
    print(f"Cuda available: {torch.cuda.is_available()}", file=sys.stdout)
    print('__CUDNN VERSION:', torch.backends.cudnn.version(), file=sys.stdout)
    print('Available devices ', torch.cuda.device_count(), file=sys.stdout)
    print('Current cuda device ', torch.cuda.current_device(), file=sys.stdout)
    print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")


if __name__=="__main__":
    test_gpu_availability()

    class Test(torch.nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)

            self.lazy_layer = torch.nn.LazyLinear(25)

        def print_trainable_parameters(self):
            print("If LazyModules are not yet initialized the total number of parameters changes after initialization.")
            trainable = sum(p.numel() for p in self.parameters() if (not isinstance(p, torch.nn.parameter.UninitializedParameter)) and p.requires_grad)
            total = sum(p.numel() for p in self.parameters() if  (not isinstance(p, torch.nn.parameter.UninitializedParameter)))
            print(f"trainable: {trainable:,} || all params: {total:,} || trainable%: {trainable/(total*100+1):.4f}")

    Test().print_trainable_parameters()
