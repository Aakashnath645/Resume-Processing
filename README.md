# 📄 Resume Screening Assistant

**Resume Screening Assistant** is a modern, AI-powered web application that analyzes resumes and matches them intelligently with job descriptions using **Google's Gemini AI model**. Built with **Streamlit** and **Python**, it helps recruiters and hiring managers streamline their candidate evaluation process.

---

## ✨ Features

- **🧠 AI-Powered Analysis**  
  Utilizes Google Gemini AI for intelligent, contextual resume parsing and matching.

- **📁 Multi-Format Support**  
  Supports resumes in PDF, DOCX, and DOC formats.

- **📦 Batch Processing**  
  Upload and evaluate multiple resumes simultaneously.

- **📊 Interactive Dashboard**  
  Real-time analytics for match scores, department-wise performance, and more.

- **📝 Custom Report Generation**  
  Generate detailed reports for individual or batch analysis in PDF/DOCX formats.

- **🎯 Role-Based Scoring**  
  Evaluates candidates based on role levels like Junior, Mid, Senior, etc.

- **🧲 ATS Compatibility**  
  Mimics ATS behavior with keyword scanning and scoring.

- **🏢 Department-Specific Analysis**  
  Customizable metrics for various departments such as Tech, HR, Marketing, etc.

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Aakashnath645/Resume-Processing.git
cd Resume-Processing
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Environment Variables

Create a `.env` file in the project root and add the following:

```env
GEMINI_API_KEY=your_google_generative_ai_api_key
```

---

## 💡 Usage

### 1. Run the Application

```bash
streamlit run Home.py
```

### 2. Open the Web Interface

- Input or upload a job description
- Select department and role level
- Upload one or more resumes
- Click **"Process Resume(s)"** to generate insights and reports

---

## 📁 Project Structure

```
Resume-Hiring/
├── .streamlit/          # Streamlit configuration
├── components/          # UI components
├── config/              # Configuration files
├── pages/               # Additional pages
├── utils/               # Utility functions
│   ├── ai_analysis.py   # AI processing logic
│   ├── extract_utils.py # Text extraction utilities
│   ├── gemini_client.py # Gemini API client
│   └── constants.py     # Constants and configurations
├── .env                 # Environment variables(excluded from git)
├── Home.py              # Main application file
└── requirements.txt     # Project dependencies
```

---

## 🔍 Feature Breakdown

### ✅ Resume Analysis

- Extracts and analyzes candidate skills, education, and experience
- Role-specific scoring for better relevance
- Cultural fit evaluation (experimental)
- ATS keyword scanning and scoring

### 📄 Reporting

- Generates individual candidate reports
- Batch analysis reports with summary metrics
- Exportable formats: PDF and DOCX
- Includes breakdowns of match scores and keyword hits

### 📈 Dashboard & Analytics

- Visual match percentages for each candidate
- Departmental breakdown of match scores
- Candidate success prediction (optional)
- Batch performance summary

---

## 🛠 Technology Stack

- **Frontend**: Streamlit
- **AI/ML**: Google Gemini AI API
- **Resume Parsing**: `PyPDF2`, `python-docx`
- **Database**: SQLite via `sqlite-utils`
- **Report Generation**: `FPDF2`, `python-docx`
- **Environment Handling**: `python-dotenv`
- **Data Manipulation**: `pandas`

---

## 🤝 Contributing

We welcome your contributions!

1. Fork this repository
2. Create a new branch:

   ```bash
   git checkout -b feature/your-feature-name
   ```

3. Make your changes and commit:

   ```bash
   git commit -m "Add: your message"
   ```

4. Push to your forked repo:

   ```bash
   git push origin feature/your-feature-name
   ```

5. Submit a Pull Request 🚀

---

## ⚙️ Requirements

- Python 3.8 or above
- Streamlit
- Google Generative AI SDK
- PyPDF2
- python-docx
- FPDF2
- pandas
- sqlite-utils
- python-dotenv

Install all dependencies via:

```bash
pip install -r requirements.txt
```

## 🙏 Acknowledgments

- **Google Gemini AI** – for powering resume analysis
- **Streamlit** – for the sleek and simple UI framework

---

## 📬 Support

Found a bug or have a feature request?  
Open an issue at: [https://github.com/Aakashnath645/Resume-Processing/issues](https://github.com/Aakashnath645/Resume-Processing/issues)

---

> 💼 Empower your recruitment with intelligent AI — faster, fairer, and smarter resume screening!
