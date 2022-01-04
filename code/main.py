import argparse
import numpy as np
import pandas as pd
from datasets import load_dataset, load_metric
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm
import time
import spacy
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import RobertaTokenizerFast
from transformers import DataCollatorForTokenClassification
from transformers import AutoTokenizer

def sentece_preprocess(sentece, span, nlp):
    """
    This function label tokens of the sentece based on its span
    Input:
    sentece [array]: the sentence that we want to labels its tokens as T(Toxic), and NT (Not toxic)
    span [str] : span in string format (Note: currently there is a bug for begining and end of that string
    for special characters of '[', and ']' characters)
    nlp: SpaCy tokenizer for english sentece
    Return:
    An array of labels for the given sentence as T or NT that can be used for true labels of token classifications
    """
    doc = nlp(sentece)
    # token.idx returns the starting span of that token
    tokens = [(token, token.idx) for token in doc]
    tokens = np.array(tokens)
    processed_token = []
    toxic_word = []
    processed_sentence = []
    span_list = []
    for item in span.replace('[', ' ').replace(']', ',').split(','):
        try:
            span_list.append(int(item))
        except:
            pass
    span_list = np.array(span_list)
    for i in range(len(tokens)):
        processed_token.append(tokens[i][0])
        if tokens[i][1] in span_list:
            toxic_word.append(tokens[i][0])
    toxic_word = np.array(toxic_word)
    for i in range(len(tokens)):
        if tokens[i][0] in toxic_word:
            #processed_sentence.append('T')
            processed_sentence.append(int(1))
        else:
            #processed_sentence.append('NT')
            processed_sentence.append(int(0))
    processed_sentence = np.array(processed_sentence)
    return processed_token, processed_sentence

def sentence_labeler(df, nlp):
    labels = []
    tokens = []
    for i in range(len(df)):
        temp_token, temp_label = sentece_preprocess(df['text'][i], df['spans'][i], nlp)
        labels.append(temp_label)
        tokens.append(temp_token)
    # converiting tokens to string
    token_str = []
    for item in tokens:
        test = []
        for element in item:
            test.append(str(element))
        token_str.append(test)
    # converiting labels to string
    label_str = []
    for item in labels:
        test_label = []
        for element in item:
            test_label.append(str(element))
        label_str.append(test_label)
    return token_str, labels


def tokenize_and_align_labels(examples, tokenizer):

    label_all_tokens = True
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples["tags"]):
        if i==1000:
          print(label)
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        #print(word_ids)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            #print(word_idx)
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

    import numpy as np

def compute_metrics(pred, metric):
    label_list = [0 , 1]
    predictions, labels = pred
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
            }

def f1(predictions, gold):
    """
    F1 (a.k.a. DICE) operating on two lists of offsets (e.g., character).
    >>> assert f1([0, 1, 4, 5], [0, 1, 6]) == 0.5714285714285714
    :param predictions: a list of predicted offsets
    :param gold: a list of offsets serving as the ground truth
    :return: a score between 0 and 1
    """
    if len(gold) == 0:
        return 1 if len(predictions)==0 else 0
    predictions1 = [tuple(l) for l in predictions]
    gold1 = [tuple(l) for l in gold]
    nom = 2*len(set(predictions1).intersection(set(gold1)))
    denom = len(set(predictions1))+len(set(gold1))
    return nom/denom

def get_tokens_from_predictions(predictions, tokenizer, tokenized_datasets):
    predicted_tokens= []
    for i in range(len(predictions)):
        prediction_arr = predictions[i]
        used_tokens = tokenizer.tokenize(tokenized_datasets['test']['text'][i])
        if(len(prediction_arr)-2 == len(used_tokens)):
            prediction_arr = prediction_arr[1:-1]
        elif(len(prediction_arr)-1 == len(used_tokens)):
            prediction_arr = prediction_arr[1:]
        elif(len(used_tokens)> len(prediction_arr)):
            for t in range(len(used_tokens)- len(prediction_arr)):
                prediction_arr.append(0)
        elif(len(used_tokens)< len(prediction_arr)):
            prediction_arr = prediction_arr[len(prediction_arr)-len(used_tokens)-1:-1]
        if(len(prediction_arr) != len(used_tokens)):
            print(len(prediction_arr), len(used_tokens))
        toxic_tokens = []
        for j in range(len(prediction_arr)):
            if(prediction_arr[j] == 1):
                if used_tokens[j][0] == "Ġ":
                    used_tokens[j] = used_tokens[j][1:]
                toxic_tokens.append(used_tokens[j])
        predicted_tokens.append(toxic_tokens)
    return predicted_tokens

def prediction_to_spans_distil_bert(predicted_tokens, tokenized_datasets):
    predicted_spans= []
    for i in range(len(predicted_tokens)):
        curr_tokens = predicted_tokens[i]
        text = tokenized_datasets['test']['text'][i]
        #print(curr_tokens)
        #print(text)
        spans = []
        for item in curr_tokens:
            idx = -1
            if item not in text:
                continue
            if len(spans) == 0:
                idx = text.index(item)
            else:
            #print(item not in text)
            #print(item)
                if(item == 'sc'):
                    text = text.lower()
                id = spans[len(spans)-1]
                if item in text[id:]:
                    idx = text.index(item, spans[len(spans)-1])
            if(idx != -1):
                for j in range(len(item)):
                    spans.append(idx)
                    idx = idx + 1
        predicted_spans.append(spans)

    return predicted_spans

def prediction_to_spans(predicted_tokens, tokenized_datasets):
    predicted_spans= []
    for i in range(len(predicted_tokens)):
        curr_tokens = predicted_tokens[i]
        text = tokenized_datasets['test']['text'][i]
        #print(curr_tokens)
        #print(text)
        spans = []
        for item in curr_tokens:
            idx = -1
            if item not in text:
                continue
            if len(spans) == 0:
                idx = text.index(item)
            else:
                idx = text.index(item, spans[len(spans)-1])
            if(idx != -1):
                for j in range(len(item)):
                    spans.append(idx)
                    idx = idx + 1
        predicted_spans.append(spans)
    return predicted_spans


def main():
    """
    This is the main function where the design logic will be applied
    """
    # Following will parse the arguments entered by the user in the command line
    parser = argparse.ArgumentParser(description='toxic span detection')
    parser.add_argument('--checkpoint', required=True, help='The path to the train model')
    parser.add_argument('--model', required=True, help='The model', choices=['RobertaTokenizerFast', 'distilbert-base-uncased'])
    parser.add_argument('--output', required=True, help='the path where the prediction will be stored')
    args = parser.parse_args()
    model = args.model
    output_path = args.output
    trained_model_path = args.checkpoint
    base_url = 'https://raw.githubusercontent.com/ipavlopoulos/toxic_spans/master/data/'
    datasets = load_dataset('csv', data_files={'train': base_url + 'tsd_train.csv', 'test': base_url + 'tsd_test.csv'})
    print(datasets)
    nlp = spacy.load("en_core_web_sm")
    print('after spacy_load')
    tokens, labels = sentence_labeler(datasets['train'], nlp)
    datasets['train'] = datasets['train'].add_column('tokens', tokens)
    datasets['train'] = datasets['train'].add_column('tags', labels)
    tokens_test, labels_test = sentence_labeler(datasets['test'], nlp)
    datasets['test'] = datasets['test'].add_column('tokens', tokens_test)
    datasets['test'] = datasets['test'].add_column('tags', labels_test)
    model_checkpoint = args.model
    model_name = model_checkpoint.split("/")[-1]
    args = TrainingArguments(
        f"{model_name}-finetuned",
        evaluation_strategy = "epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=3,
        per_device_eval_batch_size=3,
        num_train_epochs=3,
        weight_decay=0.01,
        push_to_hub=False,
    )

    if model == 'RobertaTokenizerFast':
        tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base", add_prefix_space=True)
    elif model == 'distilbert-base-uncased':
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    tokenized_datasets = datasets.map((lambda x: tokenize_and_align_labels(x, tokenizer)), batched=True)
    print('after tokenized_datasets')
    label_list = [0 , 1]
    data_collator = DataCollatorForTokenClassification(tokenizer)
    model_loaded = AutoModelForTokenClassification.from_pretrained(trained_model_path)
    metric = load_metric("seqeval")
    trainer = Trainer(
                        model_loaded,
                        args,
                        train_dataset=tokenized_datasets["train"],
                        eval_dataset=tokenized_datasets["test"],
                        data_collator=data_collator,
                        tokenizer=tokenizer,
                        compute_metrics=(lambda x: compute_metrics(x, metric))
                    )
    # trainer.train(model_path)
    predictions, labels, _ = trainer.predict(tokenized_datasets["test"])
    print('after predictions')
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
    [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
    [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(predictions, labels)
    ]

    # results = metric.compute(predictions=true_predictions, references=true_labels)
    # print('after results')
    F1_score = f1(true_predictions, true_labels)
    predicted_tokens = get_tokens_from_predictions(true_predictions, tokenizer, tokenized_datasets)
    if model == 'RobertaTokenizerFast':
        predicted_spans = prediction_to_spans(predicted_tokens, tokenized_datasets)
    elif model == 'distilbert-base-uncased':
        predicted_spans = prediction_to_spans_distil_bert(predicted_tokens, tokenized_datasets)
    print(F1_score)
    # print(predicted_spans)
    file=open(output_path,'w')
    for items in predicted_spans:
        if len(items) == 0:
            file.writelines('[]'+'\n')
        else:
            s = ''
            for i in range(len(items)):
                s = s + str(items[i]) + ','

            file.writelines('['+s[:-1]+ ']'+'\n')
    file.close()

main()