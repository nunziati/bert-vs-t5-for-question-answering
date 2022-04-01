from transformers import T5Tokenizer, T5ForConditionalGeneration, BertForQuestionAnswering, AutoTokenizer
import torch

def question_answer(model, tokenizer, question, text, device):
    
    #tokenize question and text as a pair
    input_ids = tokenizer.encode(question, text, max_length=512, truncation=True)
    
    #string version of tokenized ids
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    
    #segment IDs
    #first occurence of [SEP] token
    sep_idx = input_ids.index(tokenizer.sep_token_id)
    #number of tokens in segment A (question)
    num_seg_a = sep_idx+1
    #number of tokens in segment B (text)
    num_seg_b = len(input_ids) - num_seg_a
    
    #list of 0s and 1s for segment embeddings
    segment_ids = [0]*num_seg_a + [1]*num_seg_b
    assert len(segment_ids) == len(input_ids)
    
    #model output using input_ids and segment_ids
    output = model(torch.tensor([input_ids]).to(device), token_type_ids=torch.tensor([segment_ids]).to(device))
    
    #reconstructing the answer
    answer_start = torch.argmax(output.start_logits)
    answer_end = torch.argmax(output.end_logits)
    answer = ""
    if answer_end >= answer_start:
        answer = tokens[answer_start]
        for i in range(answer_start+1, answer_end+1):
            if tokens[i][0:2] == "##":
                answer += tokens[i][2:]
            else:
                answer += " " + tokens[i]
    
    return answer

if __name__ == '__main__':
    print("Select model:")
    print("1) Bert")
    print("2) T5")
    selection: str = int(input())
    while selection != 1 and selection != 2:
        print("Select model")
        print("1) Bert")
        print("2) T5")
        selection: str = int(input())
    
    print("Select device:")
    print("1) cpu")
    print("2) cuda")
    device: str = int(input())
    while device != 1 and device != 2:
        print("Select device:")
        print("1) cpu")
        print("2) cuda")
        device: str = int(input())

    print("Load model..")
    tokenizer = T5Tokenizer.from_pretrained("./results/t5-base/best-f1") if selection == 2 else AutoTokenizer.from_pretrained("results/bert")
    model = T5ForConditionalGeneration.from_pretrained("./results/t5-base/best-f1") if selection == 2 else BertForQuestionAnswering.from_pretrained("results/bert")

    model.to("cpu" if device == 1 else "cuda")
    model.eval()
    
    with torch.no_grad(): 
        print("Insert the context..")
        context: str = input()

        while True: 
            print("Insert the question...")
            question: str = input()
            outputs = ""
            if selection == 2:
                input_ids = tokenizer(f"question: {question}  context: {context}", return_tensors="pt").input_ids
                input_ids = input_ids.to("cpu" if device == 1 else "cuda")
                output = model.generate(input_ids)
                output = tokenizer.decode(output[0], skip_special_tokens=True)
            if selection == 1:
                output = question_answer(model, tokenizer, question, context,"cpu" if device == 1 else "cuda")
            print(f"Answer: {output}")
            print("Do you want to continue? (y/n): ")
            decision: str = input()
            if decision != "y": break
        
        
