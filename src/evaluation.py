import os

import numpy as np
import torch
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm

from src.input_features import tokenize_input


def get_key(value, dic):
    for key in dic:
        if dic[key] is value:
            return key


def evaluate(model, device, tokenizer, dev_loader, epoch, label_map, output_path, max_seq_length):
    eval_results_path = os.path.join(output_path, "eval_results.txt")
    nb_eval_steps = 0
    eval_loss = 0.0
    predictions = None
    ground_truth = None

    for batch in tqdm(dev_loader, desc="Evaluating"):
        model.eval()
        with torch.no_grad():

            input_ids, attention_mask, segment_ids, label_ids = tokenize_input(batch,
                                                                               label_map,
                                                                               tokenizer,
                                                                               max_seq_length,
                                                                               device)

            outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=segment_ids, labels=label_ids)

            # logits are always part of the output (see BertForSequenceClassification documentation), while loss is
            # only available if labels are provided. Therefore the logits are here to find on first position.
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()

        nb_eval_steps += 1

        if predictions is None:
            predictions = logits.detach().cpu().numpy()
            ground_truth = label_ids.detach().cpu().numpy()
        else:
            predictions = np.append(predictions, logits.detach().cpu().numpy(), axis=0)
            ground_truth = np.append(ground_truth, label_ids.detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps

    # remember, the logits are simply the output from the last layer, without applying an activation function (e.g.
    # sigmoid). for a simple classification this is also not necessary, we just take the index of the neuron with the
    # maximal output. To calculate the loss though, the Softmax & Cross Entropy Loss
    # is used (see BertForSequenceClassification)
    predicted_class = np.argmax(predictions, axis=1)

    simple_accuracy = (predicted_class == ground_truth).mean()
    f1 = f1_score(y_true=ground_truth, y_pred=predicted_class, average='micro')

    report = classification_report(ground_truth, predicted_class,
                                   labels=list(label_map.values()),
                                   target_names=list(label_map.keys()))

    result = {'eval_loss': eval_loss,
              'simple_accuracy': simple_accuracy,
              'f1_score': f1}

    with open(eval_results_path, "a+") as writer:
        tqdm.write("***** Eval results after epoch {} *****".format(epoch))
        writer.write("***** Eval results after epoch {} *****\n".format(epoch))
        for key in sorted(result.keys()):
            tqdm.write("{}: {}".format(key, str(result[key])))
            writer.write("{}: {}\n".format(key, str(result[key])))

        tqdm.write(report)
        writer.write(report + "\n")

    return result
