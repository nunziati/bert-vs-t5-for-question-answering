def parseForT5(example, remove_more_than_1_answer = True):
    _inputs = []
    _targets = []
    for i in range(len(example['answers']["text"])):
        _inputs.append(f"question: {example['question']}  context: {example['context']}")
        _targets.append(example['answers']["text"][i] if len(example['answers']["text"]) > 0 else "")
        if remove_more_than_1_answer: break
    return _inputs, _targets

def parseForBert(example, remove_more_than_1_answer = True):
    _inputs = []
    _targets = []
    for i in range(len(example['answers']["text"])):
        _inputs.append({
                        "question": example['question'], 
                        "context": example['context']
                        })
        _targets.append(example['answers']["text"][i] if len(example['answers']["text"]) > 0 else "")
        if remove_more_than_1_answer: break
    return _inputs, _targets