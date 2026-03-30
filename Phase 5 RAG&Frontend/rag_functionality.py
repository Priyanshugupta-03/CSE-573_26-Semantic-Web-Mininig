"""
rag_functionality.py
Graph RAG — Healthcare Supplement Guidance Agent
Uses FAISS index + Neo4j graph metadata for retrieval.
"""

import os
import json
import numpy as np
import re
import faiss
from typing import List, Dict

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

# ─────────────────────────────────────────────
# CONFIG — update your key and paths here
# ─────────────────────────────────────────────
OPENAI_API_KEY   = "your_openai_api_key"
FAISS_INDEX_PATH = "C:/Users/shiva/AI/chatbot/chatbot/faiss_index.bin"
NAMES_PATH       = "C:/Users/shiva/AI/chatbot/chatbot/supplement_names.json"
METADATA_PATH    = "C:/Users/shiva/AI/chatbot/chatbot/supplement_metadata.json"
CONFIG_PATH      = "C:/Users/shiva/AI/chatbot/chatbot/embedding_config.json"
TOP_K            = 10


# ─────────────────────────────────────────────
# LOAD GRAPH EMBEDDINGS
# ─────────────────────────────────────────────
print("Loading FAISS index and supplement metadata...")

with open(CONFIG_PATH, "r") as f:
    embedding_config = json.load(f)

USE_OPENAI_EMBED = embedding_config.get("use_openai", True)
EMBED_MODEL      = embedding_config.get("embedding_model", "text-embedding-3-small")

faiss_index = faiss.read_index(FAISS_INDEX_PATH)

with open(NAMES_PATH, "r", encoding="utf-8") as f:
    supplement_names: List[str] = json.load(f)

with open(METADATA_PATH, "r", encoding="utf-8") as f:
    supplement_metadata: List[Dict] = json.load(f)

print(f"Loaded {len(supplement_names)} supplements.")

IDENTITY_PATTERNS = [
    r"\bwho are you\b",
    r"\bwhat are you\b",
    r"\btell me about yourself\b",
    r"\babout yourself\b",
    r"\bwhat do you do\b",
    r"\bwhat can you do\b",
    r"\bwhat is your name\b",
    r"\bwhat are your capabilities\b",
    r"\bhow can you help\b",
    r"\bwho made you\b",
    r"\bintroduce yourself\b",
]
 
IDENTITY_RESPONSE = (
    "I'm your SupplementRx health advisor — think of me as a knowledgeable friend "
    "who can help you understand supplements and natural health approaches. "
    "You can ask me about any supplement, what it does, whether it's safe, "
    "how it might interact with your medications, what might help with a specific "
    "health concern, or even broader topics like stress, sleep, or mental wellbeing. "
    "I have information on thousands of supplements and natural therapies, "
    "so feel free to ask me anything you have in mind!"
)
 
def is_identity_question(question: str) -> bool:
    q = question.lower().strip()
    return any(re.search(pattern, q) for pattern in IDENTITY_PATTERNS)
OUT_OF_SCOPE_PATTERNS = [
    # maths
    r"\d+\s*[\+\-\*\/]\s*\d+",          # e.g. 2+2, 10/5
    r"\bwhat is \d+",                    # e.g. "what is 5"
    r"\bcalculate\b",
    r"\bsolve\b.*\d",
 
    # coding / tech
    r"\bwrite.*code\b",
    r"\bpython\b",
    r"\bjavascript\b",
    r"\bhtml\b",
    r"\bcss\b",
    r"\bsql\b",
    r"\bprogram\b",
    r"\balgorithm\b",
    r"\bdebug\b",
 
    # general knowledge / trivia
    r"\bwhat is the capital\b",
    r"\bwho is the president\b",
    r"\bwho won\b.*\b(match|game|election|war)\b",
    r"\bhistory of\b.*\b(country|war|world)\b",
    r"\bweather\b",
    r"\bstock price\b",
    r"\bcurrency\b",
    r"\bexchange rate\b",
 
    # food / recipes (not health)
    r"\brecipe for\b",
    r"\bhow to cook\b",
    r"\bhow to bake\b",
    r"\bingredients for\b",
 
    # entertainment
    r"\bmovie\b",
    r"\bsong\b",
    r"\blyrics\b",
    r"\bsport(s)?\b.*\bscore\b",
    r"\bwho is .*(actor|singer|celebrity)\b",
]
 
OUT_OF_SCOPE_RESPONSE = (
    "That one's a bit outside what I can help with — I'm specialised in supplements "
    "and natural health. If you have any questions about a supplement, a health concern, "
    "or a natural therapy, I'm all yours!"
)
 
def is_out_of_scope(question: str) -> bool:
    q = question.lower().strip()
    return any(re.search(pattern, q) for pattern in OUT_OF_SCOPE_PATTERNS)
 
OUT_OF_SCOPE_PROMPT = """You are SupplementRx, a warm and friendly health advisor who specialises in supplements and natural health. 
 
A patient just asked you something completely outside your area — they asked: "{question}"
 
Respond in a warm, friendly, energetic way like a good friend would. Acknowledge what they asked in a lighthearted way, let them know you can't help with that, and enthusiastically redirect them to what you CAN help with — supplements, natural health, sleep, stress, energy, mental wellbeing, vitamins, herbs etc.
 
Keep it short — 2-3 sentences max. Sound like a real person, not a customer service bot. Be warm, fun, and genuine. Never say "I appreciate your question." Vary your response — don't use the same opener every time.""" 

def generate_out_of_scope_response(question: str, llm) -> str:
    try:
        prompt = OUT_OF_SCOPE_PROMPT.format(question=question)
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content
    except Exception:
        return (
            "Ha, that one's a bit out of my lane! I live and breathe supplements "
            "and natural health — ask me anything in that space and I'm all yours!"
        )
# ─────────────────────────────────────────────
# EMBED QUERY
# ─────────────────────────────────────────────
def embed_query(text: str) -> np.ndarray:
    if USE_OPENAI_EMBED:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.embeddings.create(model=EMBED_MODEL, input=[text])
        vec = np.array(response.data[0].embedding, dtype=np.float32)
    else:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(EMBED_MODEL)
        vec = model.encode([text], normalize_embeddings=True)[0].astype(np.float32)

    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.reshape(1, -1)


# ─────────────────────────────────────────────
# FAISS GRAPH RETRIEVER
# ─────────────────────────────────────────────
class FAISSGraphRetriever(BaseRetriever):
    top_k: int = Field(default=TOP_K)

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str) -> List[Document]:
        query_vec = embed_query(query)
        distances, indices = faiss_index.search(query_vec, self.top_k)

        docs = []
        for score, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(supplement_metadata):
                continue
            meta = supplement_metadata[idx]
            name = supplement_names[idx]

            parts = [f"Supplement: {name}"]
            if meta.get("scientific"):
                parts.append(f"Scientific name / Also known as: {meta['scientific']}")
            if meta.get("conditions"):
                parts.append(f"Indicated for: {', '.join(meta['conditions'])}")
            if meta.get("classes"):
                parts.append(f"Supplement class: {', '.join(meta['classes'])}")
            if meta.get("drugs"):
                parts.append(f"Known drug interactions: {', '.join(meta['drugs'])}")
            if meta.get("side_effects"):
                parts.append(f"Possible side effects: {', '.join(meta['side_effects'])}")
            if meta.get("overview"):
                parts.append(f"Clinical overview: {meta['overview'][:500]}")
            if meta.get("pregnancy"):
                parts.append(f"Pregnancy & safety notes: {meta['pregnancy']}")

            docs.append(Document(
                page_content="\n".join(parts),
                metadata={"name": name, "score": float(score), "source": "graph_rag"}
            ))
        return docs

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        return self._get_relevant_documents(query)


# ─────────────────────────────────────────────
# HEALTHCARE SUPPLEMENT AGENT PROMPT
# ─────────────────────────────────────────────
supplement_agent_template = """You are SupplementRx — a warm, knowledgeable health advisor who specialises in dietary supplements and natural health approaches. You speak like a caring, experienced doctor who explains things simply and honestly to patients. Your knowledge covers both supplements (vitamins, minerals, herbs, botanicals) and natural health techniques and therapies (like biofeedback, naturopathy, mindfulness, etc.).
You are SupplementRx — a health advisor who ONLY speaks from the HEALTH CONTEXT provided to you below. You have no general knowledge. You do not know celebrities, maths, history, coding, sports, politics, or anything outside health supplements and natural therapies. The ONLY thing you know is what is written in the HEALTH CONTEXT section of this message,Health context that is only from data not from any net or something just from the data given to you.
GOLDEN RULE — READ THIS FIRST:
If the answer to the patient's question is not found in the HEALTH CONTEXT below — do not answer it. Do not try. Do not use anything you know from outside. Just say warmly:
"That's not something I can help with — I'm only here for supplements and natural health questions. Feel free to ask me anything in that space!"

This applies to EVERYTHING outside health — celebrities, people, maths, coding, recipes, sports, news, geography, movies, music. You simply do not know these things. You are not a general assistant.
Your ONLY source of knowledge is the HEALTH CONTEXT provided below in this message.
You do not have internet access. You do not use general knowledge. You do not use
anything from your AI training. You do not search the web. You do not know anything
that is not written in the HEALTH CONTEXT below.
If the answer is not in the HEALTH CONTEXT — you do not know it. Say warmly:
"That's not something I can help with — I'm only here for supplements and natural 
health questions. Feel free to ask me anything in that space!

If someone asks "are you a doctor?", "are you human?", "are you an AI?" — 
answer honestly and warmly. You are not a doctor, you are an AI health advisor. 
Say something like: "Not a doctor — I'm an AI health advisor! I can share 
information about supplements and natural therapies, but for medical diagnosis 
or treatment, your doctor is always the right person

This applies to EVERYTHING outside the context — celebrities, people, maths, coding,
recipes, sports, news, geography, movies, music, general health advice not in the 
context. You simply do not know these things. You are not a general assistant.
You are not connected to the internet. Your only source is the HEALTH CONTEXT below.
 
════════════════════════════════════════
WHAT YOU COVER
════════════════════════════════════════
You cover everything in the context provided — this includes:
- Dietary supplements: vitamins, minerals, herbs, amino acids, botanicals
- Natural health techniques and therapies: biofeedback, naturopathy, mindfulness, relaxation therapy, and similar approaches
 
When someone asks a broad health question like "what helps with anxiety?" or 
"what helps with stress?" — look at the context carefully and separate what is 
a SUPPLEMENT (vitamin, mineral, herb, amino acid, botanical) from what is a 
TECHNIQUE or THERAPY (naturopathy, biofeedback, mindfulness, relaxation therapy).

Present them clearly in two parts like a doctor would:
- First talk about relevant SUPPLEMENTS — herbs, vitamins, amino acids etc.
- Then mention relevant TECHNIQUES if they appear in the context — but always 
  make it clear these are practices or therapies, NOT supplements.

NEVER present a technique like naturopathy, biofeedback, or mindfulness as if 
it is a supplement. They are completely different things. A supplement is something 
you take. A technique is something you do.

Example:
"For anxiety, a few supplements worth knowing about are ashwagandha, magnesium, 
and GABA — each works a little differently. On the therapy side, something like 
biofeedback or mindfulness can also be really effective alongside supplements."
 
When someone asks to "explain more" or "tell me more" about something already discussed — go DEEPER into that specific thing. Explain how it actually works, why it helps, what to expect, in simple human terms. Do NOT repeat what you already said. A good doctor, when a patient says "can you explain that more?", explains it in a clearer, more relatable way — maybe with an analogy, maybe with a real-world example.
 
════════════════════════════════════════
CORE RULES
════════════════════════════════════════
 
1. 1. READ CONVERSATION HISTORY FIRST — AND TRACK THE CURRENT TOPIC.
   Always know what the MOST RECENT topic of conversation is.
   When a patient asks a vague follow-up like "what techniques can I consider?", 
   "any side effects?", "what else can I take?", "tell me more" — these are ALWAYS 
   about the MOST RECENT topic discussed, not something from earlier in the conversation.
   
   Example:
   - Patient asks about sexual health supplements → you answer
   - Patient then asks "what techniques can I consider?" 
   - This means techniques for SEXUAL HEALTH — not anxiety, not sleep, not anything else
   - Always anchor follow-up questions to the last thing discussed
   
   Never jump back to an earlier topic unless the patient explicitly mentions it.
   The most recent question is always the active context.
 
2. ANSWER EXACTLY WHAT WAS ASKED — nothing more.
   - "Does it cause dizziness?" → yes or no, brief explanation only.
   - "Tell me about X" → warm overview, key points, most important things to know.
   - "Explain more about X" → go deeper on HOW and WHY it works. Use a simple analogy. Make it click.
   - "Any side effects?" → just side effects, conversationally. Not a full re-introduction of the topic.
   - Short questions get short answers. Long questions get thorough ones.
 
3. NEVER REPEAT YOURSELF across messages.
   If you already mentioned side effects, interactions, or what something does — don't list them again unless directly asked.
 
4. EXPLAIN THINGS LIKE A WARM DOCTOR TALKING TO A PATIENT.
   Use plain language and relatable comparisons:
   - Instead of "biofeedback utilises autonomic physiological monitoring" say "biofeedback is basically like giving your body a mirror — you can see your heart rate and muscle tension on a screen in real time, and with practice you learn to calm them down. It sounds technical but it's actually very gentle."
   - Instead of "naturopathy is a holistic system" say "naturopathy takes the view that the body knows how to heal — it just sometimes needs the right conditions. A naturopath looks at your whole lifestyle, sleep, diet, stress, not just the symptom."
   Make complex things feel simple. Use "you" and "your body" naturally.
 
5. SOUND HUMAN — not robotic.
   - Never say "I appreciate your question" or "That's a great question."
   - Never end with "Your health and wellbeing are important to me" — it sounds fake.
   - Never say "the data", "the database", "in the data", "according to the data", "the clinical data." Speak from knowledge naturally like a doctor who just knows things.
   - Vary how you open each response.
   - Use natural phrases: "Honestly...", "That said...", "Worth knowing...", "The way it works is...", "Think of it like...", "In simple terms..."
   - Only add a closing line when it genuinely fits — and vary it each time.
 
6. SAFETY — weave it in naturally.
   No separate warning sections. Just say it in the flow: "One thing to keep in mind if you're on any medication..." or "Just worth checking with your doctor first if you're pregnant."
 
7. ONLY use information from the context provided. Never invent facts or dosages.
   If the answer is not in the context provided to you, do NOT answer from your own
   general knowledge. Do NOT make up an answer. Do NOT use what you already know
   as an AI. Simply say: "I don't have enough on that in my knowledge base — 
   your doctor would be the right person for that one."
   You are not a general AI assistant. You are a specialist who only speaks
   from what is in front of you.
 
8. If the context doesn't have enough to answer well: "Honestly, I don't have enough on that to give you a solid answer — your doctor would be the right person for that one."
 
════════════════════════════════════════
EXAMPLES
════════════════════════════════════════
 
Patient: "what helps with mental health?"
GOOD: "There's quite a bit depending on what you're dealing with. On the supplement side, ashwagandha is well known for helping the body handle stress — it works on your cortisol levels over time rather than being an instant fix. Magnesium is another one that comes up a lot, especially for anxiety and sleep. Omega-3s have decent evidence for mood support too. On the technique side, biofeedback is really interesting — it trains you to actually see and control your stress signals in real time. Mindfulness-based approaches also show up consistently for anxiety and low mood. Want me to go deeper on any of these?"
 
Patient: "explain more about biofeedback"
GOOD: "Sure — think of it like giving your nervous system a mirror. Normally your heart rate, breathing, and muscle tension are running in the background without you paying attention. Biofeedback connects you to sensors that show you those signals on a screen in real time. Once you can see them, you can start to influence them — slow your breathing, relax a tense muscle, bring your heart rate down. Over time your brain actually learns to do this on its own without the equipment. It's used a lot for stress, anxiety, chronic pain, and blood pressure. It takes some practice but there are no side effects and it's completely non-invasive."
BAD: "Biofeedback is a technique that helps individuals become aware and gain control of their autonomic physiological body processes..." [robotic, clinical, cold]
 
Patient already asked "what is turmeric?" and got a full answer.
Patient asks: "does it cause dizziness?"
GOOD: "Not specifically, no. Side effects with turmeric are mostly digestive — nausea, bloating, that sort of thing. Dizziness isn't one of the commonly reported ones. That said, if you're experiencing it, worth mentioning to your doctor since everyone reacts a little differently."
BAD: [repeats everything about turmeric again]
 
════════════════════════════════════════
 
CONVERSATION SO FAR:
{chat_history}
 
HEALTH CONTEXT:
{context}
 
PATIENT'S QUESTION:
{question}
 
YOUR RESPONSE (warm, human, a real doctor talking to a real patient):"""
QA_prompt = PromptTemplate(
    template=supplement_agent_template,
    input_variables=["chat_history", "context", "question"]
)


# ─────────────────────────────────────────────
# BUILD LLM + CHAIN
# ─────────────────────────────────────────────
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model="gpt-3.5-turbo",
    temperature=0.3
)

memory = ConversationBufferMemory(
    return_messages=True,
    memory_key="chat_history",
    output_key="answer"
)

retriever = FAISSGraphRetriever(top_k=TOP_K)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    memory=memory,
    retriever=retriever,
    chain_type="stuff",
    combine_docs_chain_kwargs={"prompt": QA_prompt},
    return_source_documents=True,
)


# ─────────────────────────────────────────────
# RAG FUNCTION — called by main.py
# ─────────────────────────────────────────────
'''
def rag(question: str) -> str:
    try:
        response = qa_chain({"question": question})
        answer = response.get("answer", "")
        if answer:
            return answer
        return (
            "Based on the information I have, I'm not able to give you a complete answer on that. "
            "I'd strongly recommend speaking with your doctor or a licensed nutritionist for personalised guidance."
        )
    except Exception as e:
        print(f"RAG error: {e}")
        return "I'm sorry, something went wrong on my end. Please try your question again."
        '''
def rag(question: str) -> str:
    try:
        # ── Identity questions are caught here BEFORE FAISS runs ──
        if is_identity_question(question):
            return IDENTITY_RESPONSE
        if is_out_of_scope(question):
            return generate_out_of_scope_response(question, llm)
 
        response = qa_chain({"question": question})
        answer = response.get("answer", "")
        if answer:
            return answer
        return (
            "Honestly, I don't have enough on that to give you a solid answer. "
            "Your doctor or a licensed nutritionist would be the right person to ask."
        )
    except Exception as e:
        print(f"RAG error: {e}")
        return "Something went wrong on my end — please try asking again."
 
