# CarCare RAG — Ready-to-Run Demo

This project is a vehicle-specific RAG demonstration.

IMPORTANT:
The included PDF manuals are SYNTHETIC educational datasets. They are not
official manufacturer manuals and must not be used for real vehicle
maintenance, repair, safety, tyre-pressure, or service decisions.

## Included demo vehicles

- Tata Nexon 2025
- Hyundai Creta 2025
- Mahindra XUV 3XO 2025
- Demo Auto Eclipse 2025

## Folder structure

CarCare_RAG_Demo/
├── app.py
├── ingest.py
├── requirements.txt
├── .env.example
├── data/
│   └── vehicles.json
├── vehicles/
│   ├── tata/nexon_2025/
│   ├── hyundai/creta_2025/
│   ├── mahindra/xuv3xo_2025/
│   └── demo_auto/eclipse_2025/
└── vector_db/

## 1. Create environment

Windows:

    python -m venv venv
    venv\Scripts\activate

## 2. Install packages

    pip install -r requirements.txt

## 3. Add Gemini key

Copy:

    .env.example

to:

    .env

Then put your Gemini API key:

    GEMINI_API_KEY=YOUR_KEY_HERE

Do NOT upload .env to GitHub.

## 4. Build the vector databases

Run:

    python ingest.py

This processes all PDFs and creates one FAISS index per vehicle.

## 5. Start the application

Run:

    streamlit run app.py

Then open the local Streamlit URL shown in the terminal.

## Example questions

- What does the battery warning mean?
- What should I do if the engine temperature warning appears?
- What tyre pressure is listed in the manual?
- When should engine oil be serviced?
- What does Eco mode do?
- What does the brake warning indicate?

## How to replace the demo data with real manuals

Later, replace the synthetic PDFs inside the vehicles/ folders with manuals
that you are legally permitted to use. Update data/vehicles.json if filenames
change, then run:

    python ingest.py

The user-facing app does NOT need a PDF upload. The manuals are prepared
offline by the project administrator and indexed once.
