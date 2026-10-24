import json, time
from llama_cpp import Llama
import os

llm = Llama(model_path=os.environ.get("MODEL", "models/qwen2.5-1.5b-instruct-q4_k_m.gguf"),
            n_ctx=1024, n_batch=128, n_threads=2,
            n_gpu_layers=0, verbose=False)
schema = {
    "anyOf": [
        {"type": "object", "properties": {
            "tool": {"const": "get_events"},
            "args": {"type": "object", "properties": {
                "type": {"enum": ["BATTERY_OVERHEAT_WARNING", "ADAS_DISENGAGEMENT", "LOW_SOC_WARNING"]},
                "range": {"enum": ["yesterday", "last_7_days", "last_14_days"]}},
                "required": ["type", "range"]}},
         "required": ["tool", "args"]},
        {"type": "object", "properties": {
            "tool": {"const": "compare"},
            "args": {"type": "object", "properties": {
                "metric": {"enum": ["energy_wh_per_km", "pack_temp_c", "speed_kmh"]},
                "segment_a": {"enum": ["city", "highway"]},
                "segment_b": {"enum": ["city", "highway"]}},
                "required": ["metric", "segment_a", "segment_b"]}},
         "required": ["tool", "args"]},
        {"type": "object", "properties": {
            "tool": {"const": "out_of_scope"},
            "args": {"type": "object", "properties": {"reason": {"type": "string"}},
                     "required": ["reason"]}},
         "required": ["tool", "args"]},
    ]
}

system = """You route questions about an electric vehicle's data to exactly one tool.
Tools:
- get_events: list warnings or events (battery overheating, ADAS disengagement, low charge) in a time range.
- compare: compare a metric between city and highway driving.
- out_of_scope: use for ANYTHING not about this vehicle's data (general knowledge, chit-chat, requests to control the car).
Reply with one JSON tool call only.

Examples:
Q: Show ADAS disengagements from yesterday.
A: {"tool": "get_events", "args": {"type": "ADAS_DISENGAGEMENT", "range": "yesterday"}}
Q: Who won the cricket world cup?
A: {"tool": "out_of_scope", "args": {"reason": "not about vehicle data"}}
Q: Unlock the doors.
A: {"tool": "out_of_scope", "args": {"reason": "vehicle control is not supported"}}
"""

questions = [
    "Show all ADAS disengagement events from yesterday.",
    "Compare energy consumption between city and highway driving.",
    "Were there any battery overheating warnings in the last 7 days?",
    "What is the capital of France?",
    "Tell me a joke.",
    "Turn off the air conditioning.",
    "How is the weather today?",
    "Any low battery warnings in the last 14 days?",
    "Please unlock the doors.",
    "Could you set the temperature to 22 degrees?",
    "Who is the president of India?",
    "What is the best pizza topping?",
    "Any overheating warnings yesterday?",
    "What is the temperature in Paris today?",
    "Show ADAS disengagement events from the last 7 days."
]
import re

CONTROL = re.compile(
    r"^\s*(please\s+)?((can|could|would|will) you\s+)?"
    r"(turn|switch|unlock|lock|open|close|start|stop|set|increase|decrease|"
    r"raise|lower|enable|disable|activate|deactivate|accelerate|brake|honk|roll)\b",
    re.I)
VEHICLE_TERMS = re.compile(
    r"\b(vehicle|car|ev|battery|soc|charge|charging|range|energy|consumption|"
    r"efficiency|adas|disengage\w*|warning\w*|event\w*|overheat\w*|temperature|"
    r"speed|trip\w*|driv\w*|highway|city|hvac|power|brak\w*|trend|compare)\b", re.I)

GROUND = {
    "BATTERY_OVERHEAT_WARNING": r"overheat|thermal|\bhot\b",
    "ADAS_DISENGAGEMENT": r"adas|disengag|driver.assist|lane",
    "LOW_SOC_WARNING": r"low|charge|\bsoc\b|state of charge",
}
RANGE_WORDS = {
    "yesterday": r"yesterday",
    "last_7_days": r"7 days|seven days|\bweek",
    "last_14_days": r"14 days|two weeks|fortnight",
}

def grounded(q, call):
    if call["tool"] != "get_events":
        return True
    a = call["args"]
    return bool(re.search(GROUND[a["type"]], q, re.I)) and \
           bool(re.search(RANGE_WORDS[a["range"]], q, re.I))

for q in questions:
    if CONTROL.match(q):
        print(f"GUARD | {q}\n   -> out_of_scope (vehicle control is not supported)\n")
        continue
    if not VEHICLE_TERMS.search(q):
        print(f"GATE  | {q}\n   -> out_of_scope (not about vehicle data)\n")
        continue
    t = time.time()
    out = llm.create_chat_completion(
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": q}],
        response_format={"type": "json_object", "schema": schema},
        temperature=0, max_tokens=100)
    text = out["choices"][0]["message"]["content"]
    call = json.loads(text)
    if not grounded(q, call):
        print(f"CHECK | {q}\n   -> rejected: arguments not supported by the question\n")
        continue
    print(f"{time.time()-t:.2f}s | {q}\n   -> {json.dumps(call)}\n")

import resource

def cgroup_gb(key):
    for line in open("/sys/fs/cgroup/memory.stat"):
        k, v = line.split()
        if k == key:
            return int(v) / 2**30
    return -1

print(f"MEM private(anon)={cgroup_gb('anon'):.2f} GB | "
      f"model file cache={cgroup_gb('file'):.2f} GB | "
      f"max RSS={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20:.2f} GB")