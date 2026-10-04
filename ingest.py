import json
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

BASE_DIR = Path(__file__).resolve().parent
CATALOG = BASE_DIR / "data" / "vehicles.json"
VECTOR_DIR = BASE_DIR / "vector_db"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def main():
    vehicles = json.loads(CATALOG.read_text(encoding="utf-8"))

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    VECTOR_DIR.mkdir(exist_ok=True)

    for vehicle in vehicles:
        vehicle_id = vehicle["vehicle_id"]
        pdf_path = BASE_DIR / vehicle["manual"]

        print(f"\nProcessing: {vehicle['brand']} {vehicle['model']} {vehicle['year']}")
        print(f"PDF: {pdf_path}")

        if not pdf_path.exists():
            print("ERROR: PDF not found. Skipping.")
            continue

        docs = PyPDFLoader(str(pdf_path)).load()

        for doc in docs:
            doc.metadata.update({
                "vehicle_id": vehicle_id,
                "brand": vehicle["brand"],
                "model": vehicle["model"],
                "year": vehicle["year"],
                "source_file": pdf_path.name,
            })

        chunks = splitter.split_documents(docs)

        print(f"Pages: {len(docs)}")
        print(f"Chunks: {len(chunks)}")

        db = FAISS.from_documents(chunks, embeddings)

        save_dir = VECTOR_DIR / vehicle_id
        db.save_local(str(save_dir))

        print(f"Saved vector index: {save_dir}")

    print("\nAll vehicle indexes created successfully.")

if __name__ == "__main__":
    main()
