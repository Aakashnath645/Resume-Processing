# 🔐 Security Policy

This document outlines the security measures and best practices implemented in the **Resume Hiring Assistant** project. Our goal is to ensure the confidentiality, integrity, and availability of candidate data and system operations.

---

## 🧱 Core Security Measures

### 1. Environment Security

- Sensitive credentials (e.g., API keys for Gemini AI) are stored in a `.env` file and **never committed to version control**.
- Use tools like `python-dotenv` to safely load environment variables.

```python
from dotenv import load_dotenv
load_dotenv()
```

---

### 2. Input Validation & Sanitization

- All user-uploaded resumes are validated for file type, size, and content before processing.
- Only `.pdf`, `.docx`, and `.doc` formats are accepted.
- Input sanitization ensures malicious scripts or payloads cannot be injected.

```python
ALLOWED_EXTENSIONS = ['pdf', 'docx', 'doc']
if uploaded_file.name.split('.')[-1] not in ALLOWED_EXTENSIONS:
    st.error("Unsupported file format.")
```

---

### 3. Secure AI Communication

- Gemini AI API calls are made over **HTTPS**.
- API keys are securely loaded from environment variables and **never logged** or exposed.
- Timeout and retry policies are in place to prevent API abuse or crashes.

---

## 🛡️ Database Security

- SQLite database is protected using proper access controls.
- SQL queries are parameterized to prevent SQL injection attacks.
- Database connections are closed after every transaction to reduce the attack surface.

```python
cursor.execute("SELECT * FROM candidates WHERE department=?", (department,))
```

---

## 📉 Error Handling

- Errors are logged securely without exposing user data.
- Users receive generic error messages to avoid information leakage.
- Backend logs are scrubbed of PII (Personally Identifiable Information).

---

## 🧪 File Handling & Storage

- Uploaded resumes are stored temporarily in memory or deleted post-processing unless explicitly saved.
- No permanent storage of raw resume files without user consent.
- All temporary files use secure, randomly generated names to avoid path traversal vulnerabilities.

---

## 🔍 Reporting Vulnerabilities

If you discover a security issue, please report it responsibly:

- Email: [aakashnath645@gmail.com](mailto:aakashnath645@gmail.com)
- Subject line: **"Security Vulnerability Report - Resume Hiring Assistant"**

We aim to respond within 72 hours and take urgent action when required.

---

## 🔄 Dependency Management

- All Python dependencies are reviewed and updated regularly.
- Security audits are performed using tools like:
  - `pip-audit`
  - `bandit`
  - `safety`

---

## 🧾 Versioning & Updates

- Security patches are released under version increments (e.g., `v1.1.1`).
- Users are advised to always use the latest release for maximum protection.



_Last updated: May 15, 2025_
