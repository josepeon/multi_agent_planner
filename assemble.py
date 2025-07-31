import json
from core.assembler import assemble_code_from_log

with open("output/task_log.json") as f:
    log_data = json.load(f)

assemble_code_from_log(log_data)