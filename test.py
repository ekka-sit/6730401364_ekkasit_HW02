import os
from dotenv import load_dotenv
from google import genai

# 1. โหลดค่า GEMINI_API_KEY จากไฟล์ .env
load_dotenv()

# 2. ตรวจสอบและสร้าง Client
if not os.getenv("GEMINI_API_KEY"):
    print("❌ Error: ไม่พบ GEMINI_API_KEY กรุณาตรวจสอบไฟล์ .env")
else:
    client = genai.Client()
    print("🔍 รายชื่อโมเดลที่ใช้ได้กับ API Key ของคุณ:\n")
    
    # 3. ดึงรายชื่อโมเดล
    for model in client.models.list():
        # แสดงเฉพาะโมเดลที่รองรับการสร้างข้อความ (generateContent)
        if hasattr(model, 'supported_actions') and "generateContent" in model.supported_actions:
            clean_name = model.name.replace("models/", "")
            print(f"• {clean_name}")