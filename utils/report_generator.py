import io
from fpdf import FPDF
from docx import Document
from docx.shared import Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
import logging

class EnhancedReport:
    """
    Generates enhanced reports in PDF and DOCX formats.
    """

    def __init__(self):
        """Initializes the EnhancedReport class."""
        pass

    def generate_detailed_report(self, analysis_results, report_type='pdf'):
        """
        Generates a detailed report in the specified format.

        Args:
            analysis_results (dict): A dictionary containing the analysis results.
            report_type (str, optional): The type of report to generate ('pdf' or 'docx'). Defaults to 'pdf'.

        Returns:
            bytes: The report in bytes format.
        """
        if report_type == 'pdf':
            return self._generate_pdf_report(analysis_results)
        elif report_type == 'docx':
            return self._generate_docx_report(analysis_results)
        else:
            raise ValueError("Invalid report type. Choose 'pdf' or 'docx'.")

    def _generate_pdf_report(self, analysis_results):
        """Generates a PDF report."""
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        for key, value in analysis_results.items():
            pdf.cell(200, 10, txt=f"{key}: {value}", ln=1)

        return pdf.output(dest='S').encode('latin-1')

    def _generate_docx_report(self, analysis_results):
        """Generates a DOCX report."""
        document = Document()
        
        # Add a title
        document.add_heading('Resume Analysis Report', level=1)

        for key, value in analysis_results.items():
            paragraph = document.add_paragraph()
            paragraph.add_run(f"{key}: {value}")

        # Save the docx file to a BytesIO object
        docx_stream = io.BytesIO()
        document.save(docx_stream)
        docx_stream.seek(0)  # Go back to the beginning of the stream

        return docx_stream.read()

    def generate_batch_pdf_report(self, batch_results):
        """
        Generates a batch PDF report containing all analysis results.

        Args:
            batch_results (list): A list of dictionaries containing analysis results.

        Returns:
            bytes: The batch report in bytes format.
        """
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        for i, results in enumerate(batch_results):
            pdf.cell(200, 10, txt=f"Report {i+1}:", ln=1)
            for key, value in results.items():
                pdf.cell(200, 10, txt=f"  {key}: {value}", ln=1)
            pdf.ln(10)  # Add some space between reports

        return pdf.output(dest='S').encode('latin-1')

    def generate_batch_docx_report(self, batch_results):
        """
        Generates a batch DOCX report containing all analysis results.

        Args:
            batch_results (list): A list of dictionaries containing analysis results.

        Returns:
            bytes: The batch report in bytes format.
        """
        document = Document()
        document.add_heading('Batch Resume Analysis Report', level=1)

        for i, results in enumerate(batch_results):
            document.add_heading(f'Report {i+1}', level=2)
            for key, value in results.items():
                paragraph = document.add_paragraph()
                paragraph.add_run(f"{key}: {value}")
            document.add_paragraph()  # Add some space between reports

        # Save the docx file to a BytesIO object
        docx_stream = io.BytesIO()
        document.save(docx_stream)
        docx_stream.seek(0)  # Go back to the beginning of the stream

        return docx_stream.read()
