# Smart Resume Screening Assistant 🔍

An AI-powered resume screening and analysis tool built with Python and Streamlit that helps recruiters and hiring managers efficiently process and evaluate resumes against job descriptions.

## Features

### Core Functionality
- 📄 Upload multiple resumes (PDF, DOCX, DOC)
- 📝 Input or upload job descriptions
- 🎯 Department and role-specific analysis
- 📊 AI-powered resume scoring and matching
- 📈 Detailed candidate assessment reports

### Analysis Capabilities
- 🤖 AI-driven resume analysis using Gemini 2.0
- 📊 ATS (Applicant Tracking System) keyword matching
- 💡 Role-specific scoring algorithms
- 📋 Comprehensive skill assessment
- 🎯 Suitability determination

### Output Features
- 📑 Detailed analysis reports in PDF and DOCX formats
- 📊 Match percentage scoring
- 👥 Candidate ranking
- 💾 Analysis storage in SQLite database
- 📈 Progress tracking with visual feedback

## Technical Requirements

```text
Python 3.8+
Streamlit
PyPDF2
python-docx
FPDF
google-generativeai
ollama(for running locally)
sqlite3
pandas
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Aakashnath645/Resume-Processing.git
cd Resume-Processing
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
GEMINI_API_KEY=your_api_key_here
```

## Usage

1. Start the application:
```bash
streamlit run Home.py
```

2. Upload job description:
   - Enter text directly or upload a file (PDF/DOCX/TXT)
   - Select department and role

3. Upload resumes:
   - Support for multiple resume uploads
   - Accepts PDF, DOCX, DOC formats

4. View results:
   - Match percentages
   - Detailed analysis
   - Download comprehensive reports

## Project Structure

```
resume-screening-assistant/
├── Home.py                    # Main application file
├── utils/
│   ├── ai_analysis.py        # AI analysis utilities
│   ├── report_generator.py   # Report generation
│   └── extract_utils.py      # Text extraction utilities
├── config/
│   └── job_roles.json        # Job role configurations
├── style.css                 # Custom styling
└── animations.html           # UI animations
```

## Features in Detail

### AI Analysis
- Technical skill evaluation
- Experience assessment
- Leadership capability analysis
- Cultural fit determination
- Role-specific scoring

### Report Generation
- Candidate overview
- Detailed skill matching
- Experience analysis
- Recommendations
- Export options (PDF/DOCX)

### Database Integration
- Analysis storage
- Historical data tracking
- Query capabilities
- Automatic backup

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Google Gemini API for AI analysis
- Streamlit for the web interface
- Open source community for various libraries used

## Support

For support, please open an issue in the GitHub repository or contact the maintenance team.
