from flask import Flask, request, jsonify
from flask_cors import CORS
import re
from datetime import datetime

app = Flask(__name__)
CORS(app) 

def get_response(user_input):
    user_input = user_input.lower()

    # Admissions
    if re.search(r'\b(admission|apply|enroll|entrance|deadline|last date|registration)\b', user_input):
        return "Admissions are open until July 30. You can apply online via our website. Entrance exam is required for B.Tech and MBA programs."

    # Courses
    elif re.search(r'\b(courses|programs|subjects|diploma|distance|duration|curriculum)\b', user_input):
        return ("We offer B.Tech (4 years), B.Sc (3 years), B.Com (3 years), MBA (2 years), MCA (2 years), and diploma courses. "
                "Distance learning is available for some programs.")

    # Fees
    elif re.search(r'\b(fees|fee structure|tuition|installment|scholarship|payment)\b', user_input):
        return ("Fee structure varies by course. Scholarships are available for merit and need-based students. "
                "Installments can be discussed during admission.")

    # Facilities
    elif re.search(r'\b(hostel|wifi|library|canteen|facilities|labs|gym|sports)\b', user_input):
        return ("We provide hostel facilities for both boys and girls, 24x7 Wi-Fi, a fully-equipped library, gym, canteen, and modern computer/science labs. Sports facilities are also available.")

    # Placement
    elif re.search(r'\b(placement|job|recruiters|salary|package|internship|hiring)\b', user_input):
        return ("Our placement cell invites companies like TCS, Infosys, and Wipro. The highest package last year was 12 LPA. Internships are also provided.")

    # Contact
    elif re.search(r'\b(contact|email|phone|call|visit|location|address)\b', user_input):
        return ("You can contact us at +91-9876543210 or email info@CodeSoftcollege.edu. We're located near Connaught Place, New Delhi. Visitors are welcome between 10 AM to 4 PM.")

    # Time
    elif re.search(r'\b(time|current time)\b', user_input):
        return f"The current time is {datetime.now().strftime('%H:%M:%S')}."

    # Faculty
    elif re.search(r'\b(faculty|teachers|professors|staff)\b', user_input):
        return "Our faculty consists of experienced professionals from IITs, IIMs, and global universities, dedicated to student success."

    # Accreditation
    elif re.search(r'\b(accreditation|approved|ugc|aicte|nba|naac)\b', user_input):
        return "Yes, CodeSoft College is UGC approved and accredited by NAAC with an 'A' grade. Our technical courses are also AICTE approved."

    # Documents
    elif re.search(r'\b(documents|required papers|certificates|marksheet|id proof)\b', user_input):
        return ("You'll need your 10th and 12th marksheets, entrance exam scorecard, passport-size photos, and ID proof during admission.")

    # Transport
    elif re.search(r'\b(transport|bus|shuttle|commute|pickup)\b', user_input):
        return "We offer bus services across major routes in the city for daily commutes."

    # Campus Life
    elif re.search(r'\b(campus life|fest|clubs|events|environment|culture)\b', user_input):
        return "Campus life is vibrant with tech clubs, cultural societies, and annual fests like 'TechNova' and 'Rhythm'."

    # Results
    elif re.search(r'\b(result|exam result|marks|grade|report card)\b', user_input):
        return "Results are declared on the student portal. You can log in with your credentials to view your marks."

    # Tech Support
    elif re.search(r'\b(technical issue|login problem|website error|app crash|password reset)\b', user_input):
        return "For technical support, email us at support@CodeSoftcollege.edu or call our helpline at +91-9123456789."

    # Greetings
    elif re.search(r'\b(hi|hello|hey|good morning|good evening)\b', user_input):
        return "Hello! Welcome to CodeSoft College. How can I assist you today?"

    # Farewell
    elif re.search(r'\b(bye|goodbye|see you|thanks|thank you)\b', user_input):
        return "Goodbye! Feel free to reach out again. Have a great day!"

    # Fallback
    else:
        return "I'm sorry, I didn't understand that. Could you please rephrase or ask a different question?"

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("message", "")
    print("User Input:", user_input)  # Log input
    response = get_response(user_input)
    print("Bot Response:", response)  # Log response
    return jsonify({"response": response})


if __name__ == "__main__":
    app.run(debug=True, port=8000)

