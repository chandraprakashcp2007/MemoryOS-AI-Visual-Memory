from __future__ import annotations
from collections import Counter
from fastapi import APIRouter
from backend.api.search import load_memories
router=APIRouter(prefix="/collections",tags=["Collections"])
RULES={"Receipts & Bills":("receipt","invoice","bill","total","gst","payment"),"Food":("food","dosa","restaurant","meal","breakfast","lunch","dinner"),"Code & Errors":("python","error","exception","traceback","gradle","android","code"),"College":("college","class","timetable","exam","assignment","kiot","student"),"Travel":("train","bus","ticket","hotel","booking","travel","flight"),"Documents":("document","identity","certificate","card","aadhaar","pan"),"Shopping":("shopping","product","price","order","amazon","flipkart"),"People":("person","people","portrait","face")}
def memory_text(memory):
    vision=memory.get("vision_analysis") if isinstance(memory.get("vision_analysis"),dict) else {}
    values=[memory.get("filename"),memory.get("category"),memory.get("summary"),memory.get("visual_description"),memory.get("ocr_text"),memory.get("keywords"),memory.get("entities"),vision.get("objects"),vision.get("visual_concepts"),vision.get("food_concepts"),vision.get("products"),vision.get("organizations"),vision.get("locations")]
    parts=[]
    for value in values:
        if isinstance(value,(list,tuple,set)): parts.extend(str(v) for v in value)
        elif value: parts.append(str(value))
    return " ".join(parts).lower()
@router.get("")
async def collections():
    memories=load_memories(); counts=Counter()
    for memory in memories.values():
        text=memory_text(memory); matched=False
        for label,terms in RULES.items():
            if any(term in text for term in terms): counts[label]+=1; matched=True
        if not matched: counts["Other"]+=1
    q={"Receipts & Bills":"receipt bill invoice","Food":"food","Code & Errors":"code error","College":"college","Travel":"travel ticket","Documents":"document","Shopping":"shopping product","People":"person","Other":"everything"}
    return {"total_memories":len(memories),"collections":[{"name":name,"count":count,"query":q.get(name,name.lower())} for name,count in counts.most_common() if count>0]}
