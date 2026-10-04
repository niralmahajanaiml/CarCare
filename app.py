import json
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI

BASE_DIR = Path(__file__).resolve().parent
CATALOG = BASE_DIR / "data" / "vehicles.json"
VECTOR_DIR = BASE_DIR / "vector_db"

load_dotenv(BASE_DIR / ".env")

st.set_page_config(
    page_title="CarCare RAG",
    page_icon="🚗",
    layout="centered",
)

st.title("🚗 CarCare RAG")
st.caption("Vehicle-specific AI assistant powered by RAG")

@st.cache_data
def load_catalog():
    return json.loads(CATALOG.read_text(encoding="utf-8"))

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

@st.cache_resource
def get_llm():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Create a .env file and add your Gemini API key."
        )

    return ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        google_api_key=api_key,
        temperature=0.1,
    )

@st.cache_resource
def load_vehicle_db(vehicle_id):
    db_path = VECTOR_DIR / vehicle_id

    if not db_path.exists():
        raise FileNotFoundError(
            f"Vector database not found for {vehicle_id}. "
            "Run: python ingest.py"
        )

    return FAISS.load_local(
        str(db_path),
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )

def answer_question(vehicle, question):
    db = load_vehicle_db(vehicle["vehicle_id"])

    docs = db.similarity_search(question, k=4)

    if not docs:
        return "I could not find relevant information in this vehicle manual.", []

    context_parts = []
    for i, doc in enumerate(docs, 1):
        page = int(doc.metadata.get("page", 0)) + 1
        context_parts.append(
            f"[SOURCE {i} | Page {page}]\n{doc.page_content}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are CarCare AI, a vehicle-owner manual assistant.

Selected vehicle:
Brand: {vehicle['brand']}
Model: {vehicle['model']}
Year: {vehicle['year']}

STRICT RULES:
1. Answer using ONLY the supplied manual context.
2. Do not invent specifications, service intervals, warning meanings,
   tyre pressures, repair procedures, or safety instructions.
3. If the answer is not supported by the context, say:
   "I couldn't find this information in the selected vehicle's manual."
4. Keep the answer practical and easy to understand.
5. Mention the relevant page number(s) from the source material.
6. This dataset is a synthetic demonstration manual, so do not present
   its values as official manufacturer specifications.

MANUAL CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""

    response = get_llm().invoke(prompt)
    answer = response.content

    sources = []
    seen = set()

    for doc in docs:
        page = int(doc.metadata.get("page", 0)) + 1
        key = (doc.metadata.get("source_file"), page)
        if key not in seen:
            seen.add(key)
            sources.append({
                "file": doc.metadata.get("source_file", "Manual"),
                "page": page,
            })

    return answer, sources

vehicles = load_catalog()

brands = sorted(set(v["brand"] for v in vehicles))
selected_brand = st.selectbox("Manufacturer", brands)

brand_vehicles = [v for v in vehicles if v["brand"] == selected_brand]
models = [v["model"] for v in brand_vehicles]
selected_model = st.selectbox("Model", models)

model_vehicles = [v for v in brand_vehicles if v["model"] == selected_model]
years = [v["year"] for v in model_vehicles]
selected_year = st.selectbox("Year", years)

vehicle = next(
    v for v in model_vehicles
    if v["year"] == selected_year
)

st.success(
    f"Selected vehicle: {vehicle['brand']} {vehicle['model']} ({vehicle['year']})"
)

question = st.text_area(
    "Ask about your vehicle",
    placeholder="Example: What does the battery warning light mean?",
    height=100,
)

if st.button("🔍 Ask CarCare AI", type="primary", use_container_width=True):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching the vehicle manual..."):
            try:
                answer, sources = answer_question(vehicle, question)

                st.subheader("🤖 Answer")
                st.write(answer)

                st.subheader("📚 Sources")
                if sources:
                    for source in sources:
                        st.write(f"- {source['file']} — Page {source['page']}")
                else:
                    st.write("No source information available.")

            except Exception as e:
                st.error(str(e))

st.divider()
st.caption(
    "Educational RAG demonstration. The included manuals are synthetic and "
    "must not be used for real vehicle maintenance or safety decisions."
)
