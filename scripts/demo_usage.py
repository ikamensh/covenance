from pathlib import Path

from pydantic import BaseModel

from covenance import ask_llm, llm_consensus, get_records, print_usage, print_call_timeline
from covenance.models import GeminiModels
from covenance.record import set_records_dir

# Save records to demo_records/ for timeline visualization testing
DEMO_RECORDS_DIR = Path(__file__).parent / "demo_records"
set_records_dir(DEMO_RECORDS_DIR)

response = ask_llm("who is David Blayne", model="gpt-4.1-nano")
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


print_usage()
print_call_timeline()
print(f"\nRecords saved to: {DEMO_RECORDS_DIR}")

for record in get_records():
    print(record.model_dump_json(indent=4))

