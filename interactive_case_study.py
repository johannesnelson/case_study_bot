# interactive_case_study.py
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")
# Define case study scenarios
case_descriptions = {
    "Market Entry": "Generate a realistic market entry scenario where a company is considering entering a new geographical or product market. Consider factors like market size, competition, entry barriers, and regulatory environment.",
    "Profitability": "Generate a realistic profitability case scenario where a company is experiencing declining profits. The scenario should involve analysis of revenue streams, cost structures, and market trends to determine why profits are falling and suggest improvements.",
    "Growth Strategy": "Generate a realistic growth strategy case scenario where a company is exploring ways to increase revenue or market share. Consider opportunities for new markets, customer segments, or product lines.",
    "Mergers and Acquisitions": "Generate a realistic M&A case scenario where a company is evaluating the potential acquisition of another company. Consider synergies, valuation, cultural fit, and integration challenges.",
}

def setup_llm(api_key, temperature=0.7):
    return ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=api_key, temperature=temperature)

def generate_case_scenario(llm, case_type):
    case_description = case_descriptions.get(case_type)
    if not case_description:
        return "Invalid case type."
    
    # Set up case generation chain
    case_scenario_prompt = PromptTemplate(
        input_variables=["case_description"],
        template=("You are a case study expert. Based on the following description, create a unique, realistic case scenario:\n\n"
                  "{case_description}\n\n"
                  "The scenario should be challenging and relevant for a case interview. "
                  "Make sure to provide sufficient details for the candidate to analyze.")
    )
    case_scenario_chain = LLMChain(llm=llm, prompt=case_scenario_prompt)
    return case_scenario_chain.run(case_description=case_description)

def run_interactive_case_study(api_key, case_type, feedback_style="Bain-style"):
    # Initialize model and memory
    llm = setup_llm(api_key)
    memory = ConversationBufferMemory()

    # Generate the case scenario
    scenario = generate_case_scenario(llm, case_type)
    print("Generated Case Scenario:", scenario)
    print("\n--- Interview ---")

    # Set up interview question chain
    interview_question_prompt = PromptTemplate(
        input_variables=["scenario", "previous_responses"],
        template=("You are an interviewer conducting a case study interview. Based on the following scenario, "
                  "ask the next question in a way that guides the candidate through analyzing the scenario. "
                  "Do not provide any candidate responses.\n\n"
                  "Scenario:\n{scenario}\n\n"
                  "Previous responses:\n{previous_responses}\n\n"
                  "Next Question:")
    )
    interview_question_chain = LLMChain(llm=llm, prompt=interview_question_prompt)

    # Loop through interactive interview
    end_interview = False
    previous_responses = ""
    while not end_interview:
        question = interview_question_chain.run(scenario=scenario, previous_responses=previous_responses)
        print("Interviewer:", question)

        user_response = input("Your response (or type 'end' to finish): ")
        if user_response.lower() == 'end':
            end_interview = True
            continue

        # Save context
        memory.save_context({"AI": question}, {"Human": user_response})
        previous_responses += f"Interviewer: {question}\nCandidate: {user_response}\n"

    # Generate feedback at the end
    feedback_prompt = PromptTemplate(
        input_variables=["feedback_style", "conversation_history"],
        template=("Provide feedback in the style of {feedback_style} for the following conversation. Analyze the user's responses, "
                  "highlight strengths (if there are no strengths, be honest but kind), areas for improvement, and suggestions for a strategic approach.\n\n"
                  "Conversation:\n{conversation_history}")
    )
    feedback_chain = LLMChain(llm=llm, prompt=feedback_prompt)
    feedback = feedback_chain.run(feedback_style=feedback_style, conversation_history=memory.buffer)
    print("\n--- Feedback ---")
    print(feedback)

if __name__ == "__main__":
    
    case_type = input("Enter the case type (Market Entry, Profitability, Growth Strategy, Mergers and Acquisitions): ")
    run_interactive_case_study(api_key, case_type)
