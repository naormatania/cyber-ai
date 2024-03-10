import re
import json
from ner_eval import Evaluator

_LINE_RE = re.compile(r"^(((\S+)(\s+)?)+) (O|(([IB])-(\S+)))$")

def read_iob_tokens(full_path):
    lines = open(full_path, 'r').readlines()
    lines = [line for line in lines if line != "O\n" and line != " O\n"]
    sent = []
    for line in lines:
        if line == "\n":
            if len(sent) != 0:
                yield sent
            sent = []
            continue
        match_ = re.match(_LINE_RE, line)
        token = match_.group(1)
        postag = match_.group(7)
        label = match_.group(8)
        sent.append((token, postag, label))
    if len(sent) != 0:
        yield sent

def sent2labels(sent, tokens_to_discard=[',','.']):
    return [(token[1], token[2]) for token in sent if token[0] not in tokens_to_discard]

def sent2label_tokens(sent, label, tokens_to_discard=[',','.']):
    return [token[0] for token in sent if token[0] not in tokens_to_discard and token[2] == label]

def change_labels(sent, mapping):
    return [(postag, label) if label not in mapping else (postag, mapping[label]) for postag, label in sent]

def label_count(sents):
    counts = {}
    for sent in sents:
        for token in sent:
            label = token[1]
            if label is None:
                continue
            if label not in counts:
                counts[label] = 1
            else:
                counts[label] = counts[label] + 1
    return counts

def write_label_count(f, sents, name):
    count = label_count(sents)
    f.write(f'{name} label count (Total - {sum(count.values())}):\n')
    f.write(json.dumps(count)+"\n")

def build_iob(sent, default='O'):
    return [f'{postag}-{label}' if label is not None else default for postag, label in sent]

true_sents = list(read_iob_tokens('DNRTI/iob.txt'))
cyner_sents = list(read_iob_tokens('test/results/iob_cyner.txt'))
secner_sents = list(read_iob_tokens('test/results/iob_secner.txt'))
flair_sents = list(read_iob_tokens('test/results/iob_flair.txt'))

true = [sent2labels(sent) for sent in true_sents]
pred_cyner = [sent2labels(sent) for sent in cyner_sents]
pred_secner = [sent2labels(sent) for sent in secner_sents]
pred_flair = [sent2labels(sent) for sent in flair_sents]

with open('test/results/label_counts.txt', 'w') as f:
    # True labels: {'Purp', 'SamFile', 'Idus', 'Way', 'OffAct', 'Area', 'Tool', 'Exp', 'O', 'Time', 'SecTeam', 'Features', 'HackOrg', 'Org'}
    write_label_count(f, true, 'True')
    # CyNER labels: {'Vulnerability', 'Indicator', 'O', 'Malware', 'System', 'Organization'}
    write_label_count(f, pred_cyner, 'CyNER')
    # SecureBERT-NER labels: {'PROT', 'IP', 'FILE', 'O', 'APT', 'MD5', 'ACT', 'VULID', 'VULNAME', 'ENCR', 'DOM', 'EMAIL', 'TOOL', 'SECTEAM',
    # 'SHA2', 'IDTY', 'OS', 'URL', 'MAL', 'LOC', 'TIME'}
    write_label_count(f, pred_secner, 'SecureBERT-NER')
    # Flair labels: {'LOC', 'O', 'PER', 'ORG', 'MISC'}
    write_label_count(f, pred_flair, 'Flair')

# Experiment1: Test CyNER/SecNER/Flair against True labels (we'll reduce all label to CyNER labels which are the most succint)
with open('test/results/exp1.txt', 'w') as f:
    true_to_cyner = {'SamFile': 'Indicator', 'Idus': 'Organization', 'Way': 'System', 
                 'OffAct': 'System', 'Tool': 'Malware', 'Exp': 'Vulnerability', 'SecTeam': 'Organization', 
                 'HackOrg': 'Organization', 'Org': 'Organization'}
    exp1_true_labels = [build_iob(change_labels(sent, true_to_cyner)) for sent in true]
    secner_to_cyner = {'PROT': 'Indicator', 'IP': 'Indicator', 'FILE': 'Indicator', 'APT': 'Organization', 'MD5': 'Indicator', 'ACT': 'System',
                    'VULID': 'Vulnerability', 'VULNAME': 'Vulnerability', 'ENCR': 'Indicator', 'DOM': 'Indicator', 'EMAIL': 'Indicator', 'TOOL': 'System',
                    'SECTEAM': 'Organization', 'SHA2': 'Indicator', 'IDTY': 'Organization', 'OS': 'System', 'URL': 'Indicator', 'MAL': 'Malware', 'LOC': 'Area', 'TIME': 'Time'}
    exp1_secner_labels = [build_iob(change_labels(sent, secner_to_cyner)) for sent in pred_secner]
    exp1_cyner_labels = [build_iob(sent) for sent in pred_cyner]
    flair_to_cyner = {'ORG': 'Organization', 'LOC': 'Area'}
    exp1_flair_labels = [build_iob(change_labels(sent, flair_to_cyner)) for sent in pred_flair]

    exp1_cyner_evaluator = Evaluator(exp1_true_labels, exp1_cyner_labels, ['Vulnerability', 'Indicator', 'Malware', 'System', 'Organization'])
    results, results_agg = exp1_cyner_evaluator.evaluate()
    f.write("################ CyNER results:\n")
    f.write(json.dumps(results, indent=2) + "\n")
    f.write(json.dumps(results_agg, indent=2) + "\n")

    exp1_secner_evaluator = Evaluator(exp1_true_labels, exp1_secner_labels, ['Vulnerability', 'Indicator', 'Malware', 'System', 'Organization', 'Time', 'Area'])
    results, results_agg = exp1_secner_evaluator.evaluate()
    f.write("################ SecNER results:\n")
    f.write(json.dumps(results, indent=2) + "\n")
    f.write(json.dumps(results_agg, indent=2) + "\n")

    exp1_flair_evaluator = Evaluator(exp1_true_labels, exp1_flair_labels, ['Organization', 'Area'])
    results, results_agg = exp1_flair_evaluator.evaluate()
    f.write("################ Flair results:\n")
    f.write(json.dumps(results, indent=2) + "\n")
    f.write(json.dumps(results_agg, indent=2) + "\n")

# Experiment2: Test SecNER against True labels on the more specific labels: APT, SECTEAM, IDTY, FILE
with open('test/results/exp2.txt', 'w') as f:
    true_to_secner = {'HackOrg': 'APT', 'SecTeam': 'SECTEAM', 'Idus': 'IDTY', 'Org': 'IDTY', 'SamFile': 'FILE'}
    exp2_true_labels = [build_iob(change_labels(sent, true_to_cyner)) for sent in true]
    exp2_secner_labels = [build_iob(sent) for sent in pred_secner]

    exp2_secner_evaluator = Evaluator(exp2_true_labels, exp2_secner_labels, ['APT', 'SECTEAM', 'IDTY', 'FILE'])
    results, results_agg = exp2_secner_evaluator.evaluate()
    f.write("################ SecNER results:\n")
    f.write(json.dumps(results, indent=2) + "\n")
    f.write(json.dumps(results_agg, indent=2) + "\n")
