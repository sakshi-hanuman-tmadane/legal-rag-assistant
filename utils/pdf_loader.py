import os
import fitz

def load_pdfs(folder_path):
    documents = []

    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, file)

            pdf = fitz.open(pdf_path)

            for page_num in range(len(pdf)):
                page = pdf.load_page(page_num)

                documents.append({
                    "text": page.get_text(),
                    "source": file,
                    "page": page_num + 1
                })

    return documents