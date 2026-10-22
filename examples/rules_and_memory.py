"""A finite memory decision; no model, data download or learned-performance claim."""
from mmla_rsi import Language, parse_rule, EvidenceMemory, MemoryBounds, Record

lang = Language(max_nodes=63, max_depth=8)
manager = parse_rule(lang, "rtma", [
    "mul", "sub", "c1", "x0", "add", "mul", "sub", "c1", "x1", "sub", "c1", "x2",
    "mul", "x1", "sub", "x4", "x2"])
memory = EvidenceMemory(MemoryBounds(slots=1), reader_identity="example-text",
                        count_tokens=len, namespace="example", source_prefix="example/")
for index, loss in enumerate((.8, .2, .9)):
    record = Record(f"illustrative-action-{index}", loss, .1, "completed", f"example/{index}", index)
    memory, receipt = memory.manage(record, language=lang, rule=manager, opportunity_cap=3)
    print({"loss": loss, "action": receipt["status"], "memory_version": memory.version})
print(memory.read()[0])
