import PyPDF2
import pytesseract
from PIL import Image

class PDFOCR:
    def __init__(self, file_path):
        self.file_path = file_path
        
    def extract_text(self):
        pdf_file = PyPDF2.PdfReader(open(self.file_path, 'rb'))
        text = ''
        for page in pdf_file.pages:
            text += page.extract_text()
        return text
    
    def extract_text_with_OCR(self):
        image = Image.open(self.file_path)
        text = pytesseract.image_to_string(image)
        return text

pdf_ocr = PDFOCR('computing.pdf')
text = pdf_ocr.extract_text()
print(text)