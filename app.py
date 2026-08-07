import os
import sys
from dotenv import load_dotenv
from PIL import Image
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

# ---------------------------------------------------------
# Load Gemini API
# ---------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------
# Configuration & Constants
# ---------------------------------------------------------
SYSTEM_INSTRUCTION = """
คุณคือ "ผู้เชี่ยวชาญด้านบัญชีส่วนตัวและกูรูด้านอาหารมืออาชีพ" 
มีบุคลิกฉลาด รอบคอบ และให้คำแนะนำที่ช่วยให้ผู้ใช้ประหยัดเงินได้จริง 
คุณสามารถช่วยบันทึกรายจ่าย คำนวณงบประมาณ และแนะนำร้านอาหารที่คุ้มค่าและเปิดบริการอยู่จริงได้
"""

MODEL_NAME = 'gemini-3.5-flash'


# Pydantic Schema for Expense Recording
class ExpenseRecord(BaseModel):
    item: str = Field(description="ชื่อรายการรายจ่าย หรือชื่อสินค้า/บริการ")
    amount: float = Field(description="จำนวนเงิน (บาท)")
    category: str = Field(description="หมวดหมู่รายจ่าย เช่น อาหาร, การเดินทาง, ช้อปปิ้ง, สาธารณูปโภค")
    note: str = Field(description="หมายเหตุเพิ่มเติม หรือข้อแนะนำสั้นๆ จากนักบัญชี")


# ---------------------------------------------------------
# Helper Functions for Features
# ---------------------------------------------------------

def get_client() -> genai.Client:
    """Initialize and return the Gemini Client."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("[Warning] GEMINI_API_KEY environment variable is not set.")
        print("Please set it using: export GEMINI_API_KEY='your_api_key'")
    return genai.Client()


def record_expense_text(client: genai.Client, text_prompt: str) -> str:
    """Feature 1a: Record expense from text (Structured Output JSON)."""
    json_config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        temperature=0.1,
        response_mime_type="application/json",
        response_schema=ExpenseRecord,
    )
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=text_prompt,
        config=json_config
    )
    return response.text


def record_expense_image(client: genai.Client, image_path: str) -> str:
    """Feature 1b: Record expense from receipt image (Multimodality + JSON)."""
    json_config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        temperature=0.1,
        response_mime_type="application/json",
        response_schema=ExpenseRecord,
    )
    try:
        receipt_image = Image.open(image_path)
    except Exception as e:
        return f"Error loading image '{image_path}': {e}"

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[receipt_image, "วิเคราะห์ยอดเงินและรายการจากใบเสร็จนี้เพื่อบันทึกรายจ่าย"],
        config=json_config
    )
    return response.text


def recommend_dining(client: genai.Client, location: str, budget: float) -> str:
    """Feature 2: Recommend dining options using Grounding (Google Search)."""
    search_config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        temperature=0.2,
        tools=[types.Tool(google_search=types.GoogleSearch())]
    )
    prompt = f"""
ช่วยแนะนำร้านอาหารใกล้เคียงพื้นที่ {location}
- งบประมาณ: ไม่เกิน {budget} บาทต่อมื้อ
- ความต้องการ: ขอร้านที่เปิดให้บริการอยู่จริงในปัจจุบัน มีชื่อร้าน ราคาโดยประมาณ และเหตุผลประกอบ
"""
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=search_config
    )
    return response.text


def calculate_budget_summary(client: genai.Client, monthly_budget: float, expenses: list[float]) -> str:
    """Feature 3: Calculate budget summary using Code Execution."""
    code_exec_config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        temperature=0.1,
        tools=[types.Tool(code_execution={})]
    )
    expenses_str = "\n".join([f"- รายจ่ายช่วงที่ {i+1}: {amt:,.2f} บาท" for i, amt in enumerate(expenses)])
    prompt = f"""
สรุปและคำนวณงบประมาณประจำเดือนให้หน่อย:
- ตั้งงบรายเดือนไว้: {monthly_budget:,.2f} บาท
{expenses_str}

ช่วยเขียนและรันโค้ดคำนวณหา:
1. ยอดรวมรายจ่ายทั้งหมดที่ใช้ไป
2. จำนวนเงินงบประมาณคงเหลือ
3. งบประมาณที่ใช้ได้ต่อวันสำหรับวันคงเหลือในเดือนนี้
"""
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=code_exec_config
    )
    return response.text


def start_chat_session(client: genai.Client):
    """Feature 4: Multi-turn Chat Session."""
    chat = client.chats.create(
        model=MODEL_NAME,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.2,
            tools=[types.Tool(google_search=types.GoogleSearch())]
        )
    )
    print("\n--- เริ่มต้นระบบแชทผู้ช่วยการเงินและแนะนำอาหาร (พิมพ์ 'exit' เพื่อจบการทำงาน) ---")
    while True:
        try:
            user_input = input("\nUser: ")
            if user_input.strip().lower() in ['exit', 'quit']:
                print("จบการสนทนา")
                break
            response = chat.send_message(user_input)
            print(f"\nAI:\n{response.text}")
        except KeyboardInterrupt:
            print("\nจบการสนทนา")
            break


def main():
    client = get_client()

    print("==================================================")
    print(" Smart Personal Finance & Dining Assistant App")
    print("==================================================")
    print("1. บันทึกรายจ่ายจากข้อความ (Structured Output JSON)")
    print("2. บันทึกรายจ่ายจากรูปใบเสร็จ (Multimodality + JSON)")
    print("3. ค้นหาร้านอาหารตามงบ (Grounding Google Search)")
    print("4. คำนวณงบประมาณประจำเดือน (Code Execution)")
    print("5. เปิดแชทโต้ตอบต่อเนื่อง (Multi-turn Chat)")
    print("6. ออกจากโปรแกรม")
    print("==================================================")

    while True:
        choice = input("\nกรุณาเลือกฟังก์ชั่น (1-6): ").strip()
        
        if choice == '1':
            text = input("ระบุรายการรายจ่าย (เช่น 'ค่าก๋วยเตี๋ยวต้มยำ 60 บาท'): ")
            res = record_expense_text(client, text)
            print("\n[ผลลัพธ์ JSON]:")
            print(res)
            
        elif choice == '2':
            img_path = input("ใส่ชื่อไฟล์หรือพาธของรูปใบเสร็จ (เช่น receipt.jpg): ")
            res = record_expense_image(client, img_path)
            print("\n[ผลลัพธ์ JSON]:")
            print(res)

        elif choice == '3':
            loc = input("ระบุสถานที่/พิกัด (เช่น สยามสแควร์ กรุงเทพฯ): ")
            budget_input = input("ระบุงบประมาณต่อมื้อ (บาท): ")
            try:
                budget = float(budget_input)
            except ValueError:
                budget = 100.0
            res = recommend_dining(client, loc, budget)
            print("\n[ผลลัพธ์แนะนำร้านอาหาร]:")
            print(res)

        elif choice == '4':
            m_budget_input = input("ระบุงบประมาณรายเดือน (กด Enter เพื่อใชัค่าเริ่มต้น 15000): ").strip()
            m_budget = float(m_budget_input) if m_budget_input else 15000.0
            
            e_str = input("ระบุรายจ่ายที่ใช้ไปแล้ว แบ่งด้วยเครื่องหมาย comma (กด Enter เพื่อใช้ค่าเริ่มต้น 3450, 2800, 4120): ").strip()
            if e_str:
                expenses = [float(x.strip()) for x in e_str.split(",") if x.strip()]
            else:
                expenses = [3450.0, 2800.0, 4120.0]
                
            res = calculate_budget_summary(client, m_budget, expenses)
            print("\n[ผลลัพธ์คำนวณงบประมาณ]:")
            print(res)

        elif choice == '5':
            start_chat_session(client)

        elif choice == '6':
            print("ขอบคุณที่ใช้บริการ!")
            sys.exit(0)
        else:
            print("ตัวเลือกไม่ถูกต้อง กรุณาลองใหม่")


if __name__ == "__main__":
    main()
