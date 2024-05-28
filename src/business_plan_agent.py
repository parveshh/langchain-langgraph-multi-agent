from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0)


class BusinessPlan(BaseModel):
    """Business plan for a given business idea"""
    product_vision: str
    main_features: List[str]
    market_research: str
    competitor_analysis: str
    challenges: str


system_prompt = """
you are an assistant who writes elaborate and stunning business plans on a given business ideas.
your plans are unique and cover the following areas:
- Product Vision
- Main Features
- Market Research
- Competitor Analysis
- Challenges

all the above areas should be answered in json format
"""

business_plan_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{business_idea}")
])

business_plan_agent = business_plan_prompt | llm.with_structured_output(
    BusinessPlan, method="json_mode")


def business_plan_node(state):
    business_idea = state["business_ideas"][state["business_idea_number"]]
    if business_idea is None:
        return END

    result = business_plan_agent.invoke({"business_idea": business_idea})
    state["business_ideas_plans"].append(result)
    state["business_idea_number"] += 1
    return state
