from covenance import ask_llm, llm_consensus, get_records
from pydantic import BaseModel

response = ask_llm("how old is David Blayne",
                   model="gpt-5-nano",
                   response_type=bool
                   )
print(response, type(response))

#
# class Evaluation(BaseModel):
#     reasoning: str
#     is_correct: bool
#
#
# eval = llm_consensus(
#     f"who is David Blayne? is this answer correct: '''{response}'''?",
#     model="gemini-2.5-flash-lite",
#     response_type=Evaluation
# )
#
#
# print(eval.model_dump_json(indent=4))
#
# for record in get_records():
#     print(record.model_dump_json(indent=4))
#
#
# # from covenance import Covenance()

