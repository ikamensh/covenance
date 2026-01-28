from covenance import ask_llm, llm_consensus, get_records
from pydantic import BaseModel
from covenance.models import GeminiModels
from covenance import print_usage

response = ask_llm("who is David Blayne", model="gpt-5-nano")
print(response)

class Evaluation(BaseModel):
    reasoning: str
    is_correct: bool


eval = llm_consensus(
    f"who is David Blayne? is this answer correct: '''{response}'''?",
    model=GeminiModels.flash_lite_25,
    response_type=Evaluation
)


print(eval.model_dump_json(indent=4))

for record in get_records():
    print(record.model_dump_json(indent=4))

print_usage()

