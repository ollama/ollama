from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama

neo4j_config = {
    'url': "bolt://localhost:7687",
    'username': "neo4j",
    'password': "neo4jneo4j"
}

      
# get graph schema from Neo4j
graph = Neo4jGraph(**neo4j_config)

question_template = """
Task:
Generate a Cypher query for a Neo4j graph database based on the provided question.
Instructions:
1. Use only the relationship types and properties defined in the schema.
2. Ensure the direction of relationships is correct as per the schema.
3. Alias entities and relationships appropriately in the query.
4. Do not include any explanations, comments, or additional text, only the Cypher query.
5. Do not perform any operations that alter the database (no CREATE, DELETE, or MERGE).
6. Use MATCH statements to find nodes and relationships.
7. Use RETURN statements to specify the query output.

Schema:
{schema}

Question:
{question}

Output:
Provide only the Cypher query that answers the question.
"""

question_prompt = PromptTemplate(
    input_variables=["schema", "question"], template=question_template
)

answer_template = """
Task:
Formulate a human-readable response based on the results of a Neo4j Cypher query.
Instructions:
1. The provided query results are authoritative. Do not doubt or attempt to correct them using your internal knowledge.
2. The response should directly address the user's question.
3. If the provided information is empty (e.g., []), respond with "I don't know the answer."
4. If the information is available, use it to construct a complete and helpful answer.
5. Assume time durations are in days unless specified otherwise.
6. Do not indicate uncertainty if there is data in the query results. Always use the available data to form the response.

Query Results:
{context}

Question:
{question}

Output:
Provide a helpful and accurate answer based on the query results.
"""

answer_prompt = PromptTemplate(
    input_variables=["context", "question"], template=answer_template
)

gaming_cypher_chain = GraphCypherQAChain.from_llm(
    cypher_llm=ChatOllama(model='llama3'),
    qa_llm=ChatOllama(model='llama3'),
    graph=graph,
    verbose=True,
    qa_prompt=answer_prompt,
    cypher_prompt=question_prompt,
    validate_cypher=True,
    top_k=200,
)

print(gaming_cypher_chain.invoke("which players are interested in the game 'FIFA 21'?"))


