from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama
from langchain.sql_database import SQLDatabase
# from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
# from langchain.agents import create_sql_agent
# from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain.agents.agent_types import AgentType
from typing import Optional, Union, Literal
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate, PromptTemplate
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.agents.output_parsers.react_single_input import MISSING_ACTION_INPUT_AFTER_ACTION_ERROR_MESSAGE
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.exceptions import OutputParserException

llm = Ollama(model="llama2")
chat_model = ChatOllama()

# Setting up the SQL Database Connection
db = SQLDatabase.from_uri("mysql://root:123456@127.0.0.1:3306/demo")


# Creating SQL Database Toolkit
toolkit = SQLDatabaseToolkit(db=db, llm=llm)


class SQLParser(ReActSingleInputOutputParser):
    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        input, output = None, None
        for line in text.splitlines():
            if line.find("\"SELECT ") >= 0:
                input = line[line.find("SELECT "):line.rfind("\"")]
            elif line.find("SELECT ") >= 0:
                input = line[line.find("SELECT "):line.rfind(";")]
            elif line.startswith("Answer: "):
                output = line[8:]
        if input is not None:
            return AgentAction(tool="sql_db_query", tool_input=input, log=text)
        if output is not None:
            return AgentFinish(return_values={"output": output}, log=text)
        raise OutputParserException(
            f"Could not parse LLM output: `{text}`",
            observation=MISSING_ACTION_INPUT_AFTER_ACTION_ERROR_MESSAGE,
            llm_output=text,
            send_to_llm=True,
        )


def create_agent(
    llm: BaseLanguageModel,
    toolkit: Optional[SQLDatabaseToolkit] = None,
    agent_type: Optional[Union[AgentType, Literal["openai-tools"]]] = None,
    verbose: bool = False,
    db: Optional[SQLDatabase] = None,
    prompt: Optional[BasePromptTemplate] = None,
):
    """simplified from create_sql_agent"""
    from langchain.agents import create_react_agent
    from langchain.agents.agent import (
        AgentExecutor,
        RunnableAgent,
    )
    from langchain.agents.agent_types import AgentType
    from langchain_community.agent_toolkits.sql.base import (SQL_PREFIX)

    toolkit = toolkit or SQLDatabaseToolkit(llm=llm, db=db)
    agent_type = agent_type or AgentType.ZERO_SHOT_REACT_DESCRIPTION
    tools = toolkit.get_tools()
    prefix = SQL_PREFIX
    prefix = prefix.format(dialect=toolkit.dialect, top_k=10)

    from langchain.agents.mrkl import prompt as react_prompt

    format_instructions = react_prompt.FORMAT_INSTRUCTIONS
    template = "\n\n".join(
        [
            react_prompt.PREFIX,
            "{tools}",
            format_instructions,
            react_prompt.SUFFIX,
        ]
    )
    prompt = PromptTemplate.from_template(template)
    agent = RunnableAgent(
        runnable=create_react_agent(
            llm, tools, prompt, output_parser=SQLParser()),
        input_keys_arg=["input"],
        return_keys_arg=["output"],
    )

    return AgentExecutor(
        name="SQL Agent Executor",
        agent=agent,
        tools=tools,
        callback_manager=None,
        verbose=verbose,
        max_iterations=1,
        max_execution_time=None,
        early_stopping_method="force",
        return_intermediate_steps=True,
    )


# Creating and Running a SQL Agent
executor = create_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)


while True:
    question = "Number of users"
    try:
        question = input("Enter your question: ")
    except:
        print("Bye!")
        break
    if question.lower() == "exit":
        break
    if len(question) <= 0:
        continue
    try:
        executor.invoke(question, handle_parsing_errors=True,
                        return_only_outputs=True)
    except Exception as e:
        print(e)
