from __future__ import annotations

import json
import os
import re
from typing import Any, Literal

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from backend.api.search import load_memories, search_memories

router = APIRouter(prefix="/assistant", tags=["AI Assistant"])
MAX_MEMORY_SOURCES = 4
MAX_OCR_PER_MEMORY = 2200

class ChatHistoryItem(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=4000)

class AssistantRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2500)
    history: list[ChatHistoryItem] = Field(default_factory=list)
    memory_id: str | None = None
    mode: Literal["auto", "memory", "web"] = "auto"

class AssistantSource(BaseModel):
    memory_id: str
    filename: str = ""
    category: str = ""
    summary: str = ""
    image_url: str = ""
    thumbnail_url: str = ""
    why_matched: list[str] = Field(default_factory=list)

_STOP = {"a","an","and","about","can","find","for","from","give","i","in","is","me","my","of","on","please","show","tell","the","this","to","what","with","image","photo"}

def _key() -> str:
    return (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()

def _model() -> str:
    return (os.getenv("MEMORYOS_ASSISTANT_MODEL","gemini-2.5-flash-lite").strip() or "gemini-2.5-flash-lite")

def _web_enabled() -> bool:
    return os.getenv("MEMORYOS_ASSISTANT_WEB","true").strip().lower() in {"1","true","yes","on"}

def _tokens(value: str) -> list[str]:
    return [x for x in re.findall(r"[a-zA-Z0-9₹$@._-]+", str(value or "").lower()) if x and x not in _STOP]

def _vision(memory: dict[str, Any]) -> dict[str, Any]:
    value = memory.get("vision_analysis")
    return value if isinstance(value, dict) else {}

def _memory_text(memory: dict[str, Any]) -> str:
    vision = _vision(memory)
    values = [
        memory.get("filename"), memory.get("category"), memory.get("summary"),
        memory.get("description"), memory.get("visual_description"), memory.get("ocr_text"),
        memory.get("entities"), memory.get("keywords"), vision.get("visual_description"),
        vision.get("objects"), vision.get("visual_concepts"), vision.get("food_concepts"),
        vision.get("entities"), vision.get("organizations"), vision.get("locations"),
        vision.get("products"), vision.get("prices"), vision.get("dates"),
    ]
    parts = []
    for value in values:
        if isinstance(value,(list,tuple,set)):
            parts.extend(str(v) for v in value if str(v).strip())
        elif value:
            parts.append(str(value))
    return " ".join(parts).lower()

def _fast_ids(message: str, memories: dict[str, Any], limit: int = 4) -> list[str]:
    tokens = _tokens(message)
    if not tokens:
        return []
    ranked = []
    for memory_id, memory in memories.items():
        text = _memory_text(memory)
        if not text:
            continue
        hits = [token for token in tokens if token in text]
        if not hits:
            continue
        ratio = len(hits) / max(1, len(tokens))
        score = ratio * 6 + sum(3 if len(token) >= 5 else 2 for token in hits)
        if " ".join(tokens) in text:
            score += 8
        if ratio >= 0.5 or len(tokens) == 1:
            ranked.append((score, memory_id))
    ranked.sort(reverse=True)
    return [mid for _, mid in ranked[:limit]]

def _source(memory_id: str, memory: dict[str, Any], why=None):
    return AssistantSource(
        memory_id=memory_id,
        filename=str(memory.get("filename") or memory.get("original_filename") or ""),
        category=str(memory.get("category") or ""),
        summary=str(memory.get("summary") or memory.get("description") or ""),
        image_url=str(memory.get("original_image_url") or memory.get("image_url") or f"/upload/file/{memory_id}"),
        thumbnail_url=str(memory.get("thumbnail_url") or f"/upload/file/{memory_id}?thumbnail=true"),
        why_matched=list(why or []),
    )

def _context(memory_id: str, memory: dict[str, Any]) -> str:
    vision = _vision(memory)
    fields = {
        "memory_id": memory_id,
        "filename": memory.get("filename"),
        "category": memory.get("category"),
        "summary": memory.get("summary") or memory.get("description"),
        "visual_description": memory.get("visual_description") or vision.get("visual_description"),
        "objects": memory.get("objects") or vision.get("objects"),
        "visual_concepts": memory.get("visual_concepts") or vision.get("visual_concepts"),
        "food_concepts": memory.get("food_concepts") or vision.get("food_concepts"),
        "entities": memory.get("entities") or vision.get("entities"),
        "organizations": vision.get("organizations"),
        "locations": vision.get("locations"),
        "products": vision.get("products"),
        "prices": vision.get("prices"),
        "dates": vision.get("dates"),
        "phone_numbers": vision.get("phone_numbers"),
        "emails": vision.get("emails"),
        "captured_at": memory.get("captured_at"),
        "ocr_text": str(memory.get("ocr_text") or "")[:MAX_OCR_PER_MEMORY],
    }
    lines = []
    for key, value in fields.items():
        if isinstance(value,(list,tuple,set)):
            rendered = ", ".join(str(v) for v in value if str(v).strip())
        else:
            rendered = str(value or "").strip()
        if rendered:
            lines.append(f"{key}: {rendered}")
    return "\n".join(lines)

def _retrieve(message: str, memory_id: str | None, allow_deep: bool):
    memories = load_memories()
    if memory_id and memory_id in memories:
        memory = memories[memory_id]
        return [_source(memory_id,memory,["Selected memory"])], _context(memory_id,memory), [memory]
    ids = _fast_ids(message, memories, MAX_MEMORY_SOURCES)
    if not ids and allow_deep:
        try:
            ids = [r.memory_id for r in search_memories(message, limit=MAX_MEMORY_SOURCES)]
        except Exception:
            ids = []
    sources, blocks, raw = [], [], []
    for mid in ids:
        memory = memories.get(mid)
        if not memory:
            continue
        sources.append(_source(mid,memory,["Retrieved memory"]))
        blocks.append("----- MEMORY -----\n"+_context(mid,memory))
        raw.append(memory)
    return sources, "\n\n".join(blocks), raw

def _receipt(memory: dict[str, Any] | None):
    if not memory:
        return None
    vision = _vision(memory)
    text = "\n".join(str(x or "") for x in [memory.get("ocr_text"),memory.get("summary"),memory.get("visual_description")])
    lower = text.lower()
    if not any(x in lower for x in ("bill","receipt","invoice","subtotal","total","tax","gst","hotel","restaurant","amount","payment")):
        return None
    lines = [re.sub(r"\s+"," ",line).strip() for line in text.splitlines() if line.strip()]
    total = re.search(r"(?i)(?:grand\s*total|net\s*(?:amount|total)|total\s*(?:amount)?|amount\s*paid)\s*[:\-]?\s*(?:₹|rs\.?|inr)?\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)", text)
    money = re.findall(r"(?i)(?:₹|rs\.?|inr)\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)", text)
    dates = re.findall(r"\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2})\b", text)
    phones = re.findall(r"(?<!\d)(?:\+91)?[6-9]\d{9}(?!\d)", re.sub(r"[\s-]+","",text))
    emails = re.findall(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", text)
    gstin = re.findall(r"\b\d{2}[A-Z]{5}\d{4}[A-Z][A-Z0-9]Z[A-Z0-9]\b", text.upper())
    address_terms = ("road","street","nagar","colony","avenue","main","salem","chennai","coimbatore","tamil nadu","district","pin")
    address = [line for line in lines if any(term in line.lower() for term in address_terms)][:4]
    orgs = vision.get("organizations") or []
    prices = vision.get("prices") or []
    return {
        "merchant": str(orgs[0]) if orgs else (lines[0][:120] if lines else None),
        "address": ", ".join(address) or None,
        "date": dates[0] if dates else ((vision.get("dates") or [None])[0]),
        "total": total.group(1) if total else (str(prices[-1]) if prices else (money[-1] if money else None)),
        "phone": phones[0] if phones else ((vision.get("phone_numbers") or [None])[0]),
        "email": emails[0] if emails else ((vision.get("emails") or [None])[0]),
        "gstin": gstin[0] if gstin else None,
    }

def _fresh(message: str) -> bool:
    q = message.lower()
    return any(x in q for x in ("today","latest","current","now","news","weather","live","recent","tomorrow","tonight"))

def _prepare(request: AssistantRequest):
    sources, context, raw = [], "", []
    if request.mode != "web":
        sources, context, raw = _retrieve(request.message,request.memory_id,allow_deep=(request.mode=="memory"))
    effective = request.mode
    if request.mode == "auto":
        effective = "memory" if sources else "web"
    if effective == "web":
        sources, context, raw = [], "", []
    receipt = _receipt(raw[0] if raw else None)
    history = "\n".join(f"{x.role.upper()}: {x.content}" for x in request.history[-8:])
    prompt = """You are MemoryOS AI. Understand multilingual and transliterated user requests. Reply in the same language and script the user used unless they explicitly request another language; for romanized/Tanglish/Hinglish-style input, keep the reply natural in that same style when practical. Use supplied private MemoryOS context only when it supports the answer. Never invent names, totals, addresses, dates, locations, IDs, or other details. For receipts/bills identify merchant/hotel, address/location, date, tax, total, phone, email and IDs when supported; otherwise say Not detected. For public questions answer normally. Be concise and useful."""
    if history: prompt += "\nRECENT CONVERSATION:\n"+history
    if context: prompt += "\nPRIVATE MEMORYOS CONTEXT:\n"+context
    if receipt: prompt += "\nLOCAL RECEIPT FIELDS:\n"+json.dumps(receipt,ensure_ascii=False)
    prompt += "\nUSER QUESTION:\n"+request.message
    return effective, sources, raw, receipt, prompt

def _fallback(sources, raw):
    if not sources:
        return "I could not find a reliable matching memory. Public/general AI needs Gemini configured."
    receipt = _receipt(raw[0] if raw else None)
    if receipt:
        rows=[("Merchant / Hotel",receipt.get("merchant")),("Address / Location",receipt.get("address")),("Date",receipt.get("date")),("Total",receipt.get("total")),("Phone",receipt.get("phone")),("Email",receipt.get("email")),("GSTIN",receipt.get("gstin"))]
        return "\n".join(f"{k}: {v or 'Not detected'}" for k,v in rows)
    first=sources[0]
    return f"I found {first.filename}. {first.summary or 'Open the source image for the original.'}"

@router.get("/status")
async def status():
    return {"configured":bool(_key()),"model":_model(),"web_grounding":_web_enabled(),"memory_rag":True,"streaming":True,"receipt_extraction":True,"multilingual":True,"reply_language":"auto","voice_frontend":True}

@router.post("/warmup")
async def warmup():
    semantic_ready=visual_ready=False
    try:
        from backend.services.semantic_embedding_service import get_embedding_service
        get_embedding_service().warmup(); semantic_ready=True
    except Exception: pass
    try:
        from backend.services.image_embedding_service import get_image_embedding_service
        get_image_embedding_service().warmup(); visual_ready=True
    except Exception: pass
    return {"semantic_ready":semantic_ready,"visual_ready":visual_ready}

@router.post("/stream")
async def stream_chat(request: AssistantRequest):
    effective,sources,raw,receipt,prompt=_prepare(request)
    def event(name,payload):
        return f"event: {name}\ndata: {json.dumps(payload,ensure_ascii=False)}\n\n"
    async def generator():
        yield event("meta",{"mode":effective,"model":_model() if _key() else None,"sources":[s.model_dump() for s in sources],"receipt":receipt})
        if not _key():
            yield event("token",{"text":_fallback(sources,raw)}); yield event("done",{"ok":True}); return
        try:
            from google import genai
            from google.genai import types
            tools=[]
            use_web=_web_enabled() and effective=="web" and _fresh(request.message)
            if use_web: tools.append(types.Tool(google_search=types.GoogleSearch()))
            kwargs={"temperature":0.1,"max_output_tokens":650}
            if tools: kwargs["tools"]=tools
            client=genai.Client(api_key=_key())
            response=client.models.generate_content_stream(model=_model(),contents=prompt,config=types.GenerateContentConfig(**kwargs))
            for chunk in response:
                piece=str(getattr(chunk,"text","") or "")
                if piece: yield event("token",{"text":piece})
            yield event("done",{"ok":True,"web_used":use_web})
        except Exception:
            yield event("error",{"message":"AI service is temporarily unavailable. Local memory search still works."})
    return StreamingResponse(generator(),media_type="text/event-stream",headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})
