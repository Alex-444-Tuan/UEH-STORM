# UEH-STORM

tạo file: secrets.toml ở trong root repo

# Set up OpenAI API key và YDC API.
OPENAI_API_KEY="your_key"

OPENAI_API_TYPE="openai"

YDC_API_KEY="you.com_key"
vào link này để lấy YDC_API_KEY
https://api.you.com/

# SETUP:

tạo môi trường

python -m venv [ ten_moi_truong ]

source [ ten_moi_truong ]/bin/activate

pip install -r requirements.txt

chainlit run app.py

