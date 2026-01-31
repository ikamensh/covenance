from covenance.models import GeminiModels
from covenance import Covenance
from pydantic import BaseModel


client1 = Covenance(label="question client")

response = client1.ask_llm("who is David Blayne", model="gpt-4.1-nano")
print(response)


client2 = Covenance(label="consensus client")
class Evaluation(BaseModel):
    reasoning: str
    is_correct: bool


eval = client2.llm_consensus(
    f"who is David Blayne? is this answer correct: '''{response}'''?",
    model=GeminiModels.flash_lite_25,
    response_type=Evaluation
)


print(eval.model_dump_json(indent=4))

# for record in get_records():
#     print(record.model_dump_json(indent=4))

client1.print_usage()
client2.print_usage()

# from covenance import Covenance()

