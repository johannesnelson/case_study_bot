# interactive_case_study_app.py
import os
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os

# Set up the OpenAI API key
load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")


# Define case study scenarios
case_descriptions = {
    "Market Entry": "Generate a realistic market entry scenario where a company is considering entering a new geographical or product market. Consider factors like market size, competition, entry barriers, and regulatory environment.",
    "Profitability": "Generate a realistic profitability case scenario where a company is experiencing declining profits. The scenario should involve analysis of revenue streams, cost structures, and market trends to determine why profits are falling and suggest improvements.",
    "Growth Strategy": "Generate a realistic growth strategy case scenario where a company is exploring ways to increase revenue or market share. Consider opportunities for new markets, customer segments, or product lines.",
    "Mergers and Acquisitions": "Generate a realistic M&A case scenario where a company is evaluating the potential acquisition of another company. Consider synergies, valuation, cultural fit, and integration challenges.",
}

# Initialize the language model
def setup_llm(api_key, temperature=0.7):
    return ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=api_key, temperature=temperature)

# Function to generate case scenario
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

# Initialize Streamlit app
st.title("Interactive Case Study Web App")
case_type = st.selectbox("Select Case Type", list(case_descriptions.keys()))
feedback_style = st.selectbox("Select Feedback Style", ["Bain-style", "McKinsey-style", "BCG-style"])

# Initialize session state variables to manage the flow
if "scenario" not in st.session_state:
    st.session_state["scenario"] = ""
    st.session_state["previous_responses"] = ""
    st.session_state["end_interview"] = False
    st.session_state["question"] = ""
    st.session_state["conversation"] = []
    st.session_state["llm"] = setup_llm(api_key)
    st.session_state["memory"] = ConversationBufferMemory()

# Step 1: Generate the case scenario once when starting the interview
if st.button("Start Case Study") and not st.session_state["scenario"]:
    st.session_state["scenario"] = generate_case_scenario(st.session_state["llm"], case_type)
    st.write("## Generated Case Scenario:")
    st.write(st.session_state["scenario"])

# Display the scenario if already generated
if st.session_state["scenario"]:
    st.write("## Scenario:")
    st.write(st.session_state["scenario"])

    # Set up question generation chain
    interview_question_prompt = PromptTemplate(
        input_variables=["scenario", "previous_responses"],
        template=("You are an interviewer conducting a case study interview. Based on the following scenario, "
                  "ask the next question in a way that guides the candidate through analyzing the scenario. "
                  "Do not provide any candidate responses.\n\n"
                  "Scenario:\n{scenario}\n\n"
                  "Previous responses:\n{previous_responses}\n\n"
                  "Next Question:")
    )
    interview_question_chain = LLMChain(llm=st.session_state["llm"], prompt=interview_question_prompt)

    # Step 2: Generate and display the next question
    if not st.session_state["end_interview"]:
        if not st.session_state["question"]:  # Generate the first question
            st.session_state["question"] = interview_question_chain.run(
                scenario=st.session_state["scenario"], 
                previous_responses=st.session_state["previous_responses"]
            )
        st.write("**Interviewer:**", st.session_state["question"])

        # Get user input for the response
        user_response = st.text_input("Your response (leave empty and press Enter to end):")

        # Process response if provided
        if user_response:
            # Save conversation context
            st.session_state["memory"].save_context({"AI": st.session_state["question"]}, {"Human": user_response})
            st.session_state["previous_responses"] += f"Interviewer: {st.session_state['question']}\nCandidate: {user_response}\n"
            st.session_state["conversation"].append((st.session_state["question"], user_response))

            # Prepare for the next question or end the interview
            st.session_state["question"] = interview_question_chain.run(
                scenario=st.session_state["scenario"], 
                previous_responses=st.session_state["previous_responses"]
            )
        elif user_response == "":
            st.session_state["end_interview"] = True

# Step 3: Generate feedback once the interview ends
if st.session_state["end_interview"]:
    feedback_prompt = PromptTemplate(
        input_variables=["feedback_style", "conversation_history"],
        template=("Provide feedback in the style of {feedback_style} for the following conversation. Analyze the user's responses, "
                  "highlight strengths (if there are no strengths, be honest but kind), areas for improvement, and suggestions for a strategic approach.\n\n"
                  "Conversation:\n{conversation_history}")
    )
    feedback_chain = LLMChain(llm=st.session_state["llm"], prompt=feedback_prompt)
    feedback = feedback_chain.run(
        feedback_style=feedback_style, 
        conversation_history=st.session_state["memory"].buffer
    )
    st.write("\n--- Feedback ---")
    st.write(feedback)
