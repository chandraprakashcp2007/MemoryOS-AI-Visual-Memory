from backend.api.search import load_memories, search_memories
queries=["food","dosa","black bike","blue shirt","light","receipt 850","python error","android gradle error","college timetable","yesterday outside photo"]
memories=load_memories();print("INDEXED MEMORIES:",len(memories))
if not memories: print("Connect Gallery once, then run this test again.")
else:
    for q in queries:
        try:
            r=search_memories(q,limit=5);print("\n",q,":",len(r))
            for x in r[:3]: print(" -",x.filename,"|","; ".join(x.why_matched[:2]))
        except Exception as exc: print(q,"ERROR",exc)
