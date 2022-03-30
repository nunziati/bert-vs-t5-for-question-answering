def parse(example, remove_more_than_1_answer = True):
    _inputs = []
    _targets = []
    for i in range(len(example['answers'])):
        _inputs.append(f"question: {example['question']}  context: {example['plot']}")
        _targets.append(example['answers'][i] if len(example['answers']) > 0 else "")
        if remove_more_than_1_answer: break
    return _inputs, _targets