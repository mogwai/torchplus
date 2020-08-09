from harrycb/python3.7torch1.5.0
workdir /app
copy requirements.txt requirements.txt
run pip install -r requirements.txt pytest
copy . .
run pytest tests