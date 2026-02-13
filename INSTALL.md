# Project Installation Guide

Follow these steps to set up and run the project on your local machine.

## Prerequisites
- Python 3.x installed on your system.

## 1. Create a Virtual Environment
It is recommended to use a virtual environment to manage dependencies.

**Windows:**
```bash
python -m venv start
```

**Mac/Linux:**
```bash
python3 -m venv start
```

## 2. Activate the Virtual Environment

**Windows:**
```bash
.\start\Scripts\activate
```

**Mac/Linux:**
```bash
source start/bin/activate
```

## 3. Install Dependencies
Install all required packages from the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

## 4. Run the Server
Once installation is complete, you can start the Django development server.

```bash
python manage.py runserver
```

Open your browser and navigate to `http://127.0.0.1:8000/` to view the application.

## ⚠️ Important Note regarding Data Files
This project relies on large dataset files (`.npy`) and a trained model (`.pth`) located in `Sea_dataset/Sea_dataset/`. 

**These files are NOT included in the Git repository due to their size.** 

If you are cloning this repository to a new machine, you must manually copy the `Sea_dataset` folder from your original source or download it from [INSERT LINK IF AVAILABLE] and place it in the project root. The application **will not work** without these files.
