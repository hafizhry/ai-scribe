import os
from datetime import datetime
from fpdf import FPDF
import streamlit as st
from typing import Optional

class SummaryPDF(FPDF):
    def header(self):
        """Add header to each page"""
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'MedAI Scribe - Summary Report', 0, 1, 'C')
        # Line break
        self.ln(10)

    def footer(self):
        """Add footer to each page"""
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        # Add page number
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title):
        """Add a chapter title"""
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(4)

    def chapter_body(self, body):
        """Add the chapter body text"""
        self.set_font('Arial', '', 11)
        # Handle markdown-style bold text
        parts = body.split('**')
        is_bold = False
        for part in parts:
            if part.strip():
                self.set_font('Arial', 'B' if is_bold else '')
                self.multi_cell(0, 10, part)
            is_bold = not is_bold
        self.ln()

def export_summary_to_pdf(
    transcription: str,
    summary: str,
    template_type: str,
    export_dir: str = "summary_export"
) -> Optional[str]:
    """
    Export the transcription and summary to a PDF file.
    
    Args:
        transcription: The transcribed text
        summary: The generated summary
        template_type: The type of template used
        export_dir: Directory to save the PDF
        
    Returns:
        str: Path to the generated PDF file, or None if export fails
    """
    try:
        # Create export directory if it doesn't exist
        os.makedirs(export_dir, exist_ok=True)
        
        # Initialize PDF
        pdf = SummaryPDF()
        
        # Add metadata
        pdf.set_title('MedAI Scribe Summary Report')
        pdf.set_author('MedAI Scribe')
        
        # First page - Summary
        pdf.add_page()
        pdf.chapter_title(f"Summary ({template_type.upper()} Format)")
        pdf.chapter_body(summary)
        
        # Second page - Transcription
        pdf.add_page()
        pdf.chapter_title("Full Transcription")
        pdf.chapter_body(transcription)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"MedAI_summary_{timestamp}.pdf"
        file_path = os.path.join(export_dir, filename)
        
        # Save PDF
        pdf.output(file_path)
        return file_path
        
    except Exception as e:
        st.error(f"Error generating PDF: {str(e)}")
        return None

def get_pdf_download_button(file_path: str):
    """Create a download button for the PDF file."""
    try:
        with open(file_path, "rb") as f:
            pdf_data = f.read()
        
        filename = os.path.basename(file_path)
        st.download_button(
            label="Download Summary Report",
            data=pdf_data,
            file_name=filename,
            mime="application/pdf",
            key="pdf_download"
        )
        return True
    except Exception as e:
        st.error(f"Error creating download button: {str(e)}")
        return False