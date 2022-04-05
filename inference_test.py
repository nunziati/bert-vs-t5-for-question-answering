from transformers import T5Tokenizer, T5ForConditionalGeneration, BertForQuestionAnswering, AutoTokenizer, MT5ForConditionalGeneration
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
    selection: str = -1
    while selection != 1 and selection != 2:
        print("1) Bert")
        print("2) T5")
        selection: str = int(input())
    
    print("Select device:")
    device: str = -1
    while device != 1 and device != 2:
        print("1) cpu")
        print("2) cuda")
        device: str = int(input())
        
    print("Input types:")
    type: str = -1
    while type != 1 and type != 2:
        print("1) Demo")
        print("2) Free Question")
        type: str = int(input())

    print("Loading model..")
    tokenizer = T5Tokenizer.from_pretrained("./results/t5-base/best-f1") if selection == 2 else AutoTokenizer.from_pretrained("results/bert")
    model = T5ForConditionalGeneration.from_pretrained("./results/t5-base/best-f1") if selection == 2 else BertForQuestionAnswering.from_pretrained("results/bert")
    print("Model loaded.")
    model.to("cpu" if device == 1 else "cuda")
    model.eval()
    
    with torch.no_grad(): 
        if type == 2:
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
        if type == 1:
            context = """Vatican City (/ˈvætɪkən/ (audio speaker iconlisten)), officially the Vatican City State (Italian: Stato della Città del Vaticano;[e] Latin: Status Civitatis Vaticanae),[f][g] is an independent city-state and enclave surrounded by Rome, Italy.[11][12] The Vatican City State, also known simply as the Vatican, became independent from Italy with the Lateran Treaty (1929), and it is a distinct territory under "full ownership, exclusive dominion, and sovereign authority and jurisdiction" of the Holy See, itself a sovereign entity of international law, which maintains the city state's temporal, diplomatic, and spiritual independence.[h][13] With an area of 49 hectares (121 acres)[b] and a population of about 825,[8] it is the smallest state in the world by both area and population.[14] As governed by the Holy See, the Vatican City State is an ecclesiastical or sacerdotal-monarchical state (a type of theocracy) ruled by the pope who is the bishop of Rome and head of the Catholic Church.[3][15] The highest state functionaries are all Catholic clergy of various national origins. After the Avignon Papacy (1309–1377) the popes have mainly resided at the Apostolic Palace within what is now Vatican City, although at times residing instead in the Quirinal Palace in Rome or elsewhere."""
            #question: str = "How much eyes has Donal Duck?"
            questions: str = ["What is the Holy See?","What is the political situation of Vatican City?","Who is the pope?","Is the pope living in Vatican City?","Is the pope living in America?"]
            for question in questions:
                print(f"Question: {question}")
                if selection == 2:
                    input_ids = tokenizer(f"question: {question}  context: {context}", return_tensors="pt").input_ids
                    input_ids = input_ids.to("cpu" if device == 1 else "cuda")
                    output = model.generate(input_ids)
                    output = tokenizer.decode(output[0], skip_special_tokens=True)
                if selection == 1:
                    output = question_answer(model, tokenizer, question, context,"cpu" if device == 1 else "cuda")
                print(f"Answer: {output}")
        
