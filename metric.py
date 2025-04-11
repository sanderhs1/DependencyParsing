from statistics import mean
from typing import List, Tuple
import json


def pos_accuracy(gold: List[str], prediction: List[str]) -> float:
    if len(gold) != len(prediction):
        return 0.0
    return mean(1 if g == p else 0 for g, p in zip(gold, prediction))


def unlabeled_attachment_score(gold: List[Tuple[int, int]], prediction: List[Tuple[int, int]]) -> float:
    gold = set(tuple(dependency) for dependency in gold)
    prediction = set(tuple(dependency) for dependency in prediction)

    precision = len(gold & prediction) / len(prediction)
    recall = len(gold & prediction) / len(gold)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    return f1


def labeled_attachment_score(gold: List[Tuple[int, int, str, str]], prediction: List[Tuple[int, int, str, str]]) -> float:
    gold = set(gold)
    prediction = set(prediction)

    precision = len(gold & prediction) / len(prediction)
    recall = len(gold & prediction) / len(gold)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    return f1


def create_labeled_dependencies(pos_tags: List[str], dependencies: List[Tuple[int, int]]) -> List[Tuple[int, int, str, str]]:
    assert min(min(head, dependency) for head, dependency in dependencies) >= 0, "Dependency index out of bounds"
    assert max(max(head, dependency) for head, dependency in dependencies) <= len(pos_tags), "Dependency index out of bounds"
    
    pos_tags = ["ROOT"] + pos_tags
    
    return [(head, dependency, pos_tags[head], pos_tags[dependency]) for head, dependency in dependencies]


def sentence_metrics(sentence_gold, sentence_prediction):
    pos_acc = pos_accuracy(sentence_gold["pos_tags"], sentence_prediction["pos_tags"])
    uas = unlabeled_attachment_score(sentence_gold["dependencies"], sentence_prediction["dependencies"])
    las = labeled_attachment_score(
        create_labeled_dependencies(sentence_gold["pos_tags"], sentence_gold["dependencies"]),
        create_labeled_dependencies(sentence_prediction["pos_tags"], sentence_prediction["dependencies"])
    )

    return {
        "POS accuracy": pos_acc,
        "Unlabeled attachment score": uas,
        "Labeled attachment score": las
    }


def dataset_metrics(gold_path: str, prediction_path: str, verbose=True):
    gold_sentences = [json.loads(line) for line in open(gold_path)]
    prediction_sentences = [json.loads(line) for line in open(prediction_path)]

    assert len(gold_sentences) == len(prediction_sentences), "Number of sentences do not match"

    metrics = [sentence_metrics(gold, prediction) for gold, prediction in zip(gold_sentences, prediction_sentences)]

    metrics = {
        "POS accuracy": mean(metric["POS accuracy"] for metric in metrics),
        "Unlabeled attachment score": mean(metric["Unlabeled attachment score"] for metric in metrics),
        "Labeled attachment score": mean(metric["Labeled attachment score"] for metric in metrics)
    }

    if verbose:
        print("METRICS")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.2%}")

    return metrics