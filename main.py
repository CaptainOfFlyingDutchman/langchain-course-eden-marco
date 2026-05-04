from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from tavily import TavilyClient

load_dotenv()

tavily = TavilyClient()


@tool
def search(query: str) -> dict:
    """
    Tool that searches over internet
    :param query: The query to search for
    :return: The search result
    """
    print(f"Searching for {query}")
    return tavily.search(query=query)


llm = ChatOpenAI(model="gpt-5")
tools = [search]
agent = create_agent(model=llm, tools=tools)


def main():
    result = agent.invoke(
        input={
            "messages": HumanMessage(
                content="What is the weather in Faridabad, Haryana?"
            )
        }
    )
    print(result)


if __name__ == "__main__":
    main()
