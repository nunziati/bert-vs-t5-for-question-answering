from datasets import load_dataset

squad = load_dataset("squad")

from transformers import AutoTokenizer
from transformers import pipeline
from transformers import AutoModelForQuestionAnswering

model = AutoModelForQuestionAnswering.from_pretrained("./results/checkpoint-8000")

tokenizer = AutoTokenizer.from_pretrained("./results/checkpoint-8000")

pippo = pipeline("question-answering", tokenizer=tokenizer, model=model)

context = """The Sanremo Music Festival, officially the Italian Song Festival (Italian: Festival della canzone italiana) and commonly known as just Festival di Sanremo (Italian pronunciation: [sanˈremo]), is the most popular Italian song contest and awards ceremony, held annually in the city of Sanremo, Liguria.[1][better source needed] It is the longest-running annual TV music competition in the world on a national level[2] and it is also the basis and inspiration for the annual Eurovision Song Contest. Unlike other awards in Italy, the Sanremo Music Festival is a competition for new songs, not an award to previous successes (like the Premio Regia Televisiva for television, the Premio Ubu for stage performances, and the Premio David di Donatello for motion pictures)."""
question = "What is Sanremo Music Festival?"

print(pippo(question=question, context=context))