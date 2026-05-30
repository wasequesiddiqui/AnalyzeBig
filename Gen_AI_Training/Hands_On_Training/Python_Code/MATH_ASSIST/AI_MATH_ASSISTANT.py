# ================================
# IMPORTS
# ================================

import os
import re
from typing import Dict, Union

from dotenv import load_dotenv

# LangChain core
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage

# Wikipedia tool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

# ================================
# ENV SETUP
# ================================

load_dotenv()

# Initialize LLM
llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY")
)

# ================================
# MATH TOOLS
# ================================

@tool
def sum_numbers(inputs: str) -> Dict[str, Union[float, str]]:
    """Extract numbers from text and return their sum."""
    nums = re.findall(r'-?\d+(?:\.\d+)?', inputs)
    if not nums:
        return {"result": "No numbers found"}
    return {"result": sum(float(n) for n in nums)}


@tool
def multiply_numbers(inputs: str) -> Dict[str, Union[float, str]]:
    """Extract numbers and return their product."""
    nums = re.findall(r'-?\d+(?:\.\d+)?', inputs)
    if not nums:
        return {"result": "No numbers found"}

    result = 1
    for n in nums:
        result *= float(n)

    return {"result": result}


@tool
def divide_numbers(inputs: str) -> Dict[str, Union[float, str]]:
    """Extract numbers and divide them sequentially."""
    nums = re.findall(r'-?\d+(?:\.\d+)?', inputs)

    if len(nums) < 2:
        return {"result": "Need at least two numbers"}

    nums = [float(n) for n in nums]
    result = nums[0]

    for n in nums[1:]:
        if n == 0:
            return {"result": "Division by zero"}
        result /= n

    return {"result": result}


# ================================
# WIKIPEDIA TOOL
# ================================

wiki_api = WikipediaAPIWrapper()
wikipedia_tool = WikipediaQueryRun(api_wrapper=wiki_api)

# ================================
# TOOL REGISTRATION
# ================================

tools = [
    sum_numbers,
    multiply_numbers,
    divide_numbers,
    wikipedia_tool   # Only external tool now
]

tool_map = {tool.name: tool for tool in tools}

# Bind tools to LLM
llm_with_tools = llm.bind_tools(tools)

# ================================
# EXECUTION LOOP
# ================================

# Try mixed queries
query = "In 2023, the US GDP was approximately $27.72 trillion, while Canada's was around $2.14 trillion and Mexico's was about $1.79 trillion what is the total. What is the individual GDP of Canada and Mexico as percentage of the US GDP? Who is the current president of the United States Mexico and Canada list their educational background? what is the summer average temperature in each country?"

messages = [HumanMessage(content=query)]

while True:
    response = llm_with_tools.invoke(messages)

    # Add assistant response
    messages.append(response)

    # If no tool calls → final answer
    if not response.tool_calls:
        print("\n✅ Final Answer:")
        print(response.content.strip())
        break

    # Execute tool calls
    for tool_call in response.tool_calls:
        tool_name = tool_call["name"]
        print(f"\n🔧 Executing tool: {tool_name} with args: {tool_call['args']}")
        tool_args = tool_call["args"]
        print(f"Tool args: {tool_args}")

        tool_function = tool_map[tool_name]
        print(f"Invoking tool function: {tool_function}")
        try:
            # Works for most tools
            result = tool_function.invoke(tool_args)
        except:
            # Wikipedia expects "query"
            result = tool_function.invoke(tool_args.get("query", ""))

        messages.append(
            ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"]
            )
        )