# Video Face Embedding UI - Professional Analysis & Improvement

This document contains the installation and configuration steps for the project.

---

## 🛠 Installation and Configuration

You can prepare the project by running the following commands sequentially in your terminal. All steps are provided in a copy-paste format.

### 1. Virtual Environment Operations
```bash
# Create virtual environment
python -m venv env 

# Activate virtual environment (Windows)
env/Script/activate

# Install required packages
pip install -r requirements.txt

````

### 2. Static Package Installation
```bash
# Install static packages
python manage.py download_assets
```

### 3. Database Operations
```bash
# Create database
python manage.py migrate

python load_data.py
```
