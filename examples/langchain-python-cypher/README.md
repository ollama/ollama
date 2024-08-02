## About

Here, you'll learn how to generate Cypher queries using the Llama3 model to query a Neo4j graph database. The process begins with loading data into the Neo4j database through the Neo4j Admin console. After this, graph nodes and relationships are established using Cypher commands provided in the data_cypher file. Subsequently, the GraphCypherQAChain and Llama3 model, combined with custom prompt engineering, are employed to generate and parse queries into a human-readable format. This approach allows you to extract insights from the graph data, which, in this scenario, pertains to a hypothetical gaming company.

Sample csv files are provided in the data folder.

For the question in this script following cypher query is generated and response also formatted into human readable Q&A:
```
Generated Cypher:
MATCH (g:Gamer)-[:PURCHASED]-(p:Purchase)-[:CONTAINS]->(game:Game) WHERE game.Game_Name = "FIFA 21" RETURN g.Gamer_ID, g.First_Name, g.Last_Name;
Full Context:
[{'g.Gamer_ID': '10', 'g.First_Name': 'Patricia', 'g.Last_Name': 'Martinez'}, {'g.Gamer_ID': '8', 'g.First_Name': 'Laura', 'g.Last_Name': 'Miller'}, {'g.Gamer_ID': '2', 'g.First_Name': 'Jane', 'g.Last_Name': 'Smith'}, {'g.Gamer_ID': '3', 'g.First_Name': 'Robert', 'g.Last_Name': 'Brown'}, {'g.Gamer_ID': '9', 'g.First_Name': 'James', 'g.Last_Name': 'Davis'}, {'g.Gamer_ID': '1', 'g.First_Name': 'John', 'g.Last_Name': 'Doe'}, {'g.Gamer_ID': '8', 'g.First_Name': 'Laura', 'g.Last_Name': 'Miller'}, {'g.Gamer_ID': '10', 'g.First_Name': 'Patricia', 'g.Last_Name': 'Martinez'}, {'g.Gamer_ID': '6', 'g.First_Name': 'Emily', 'g.Last_Name': 'Jones'}]

> Finished chain.
{'query': "which players are interested in the game 'FIFA 21'?", 'result': 'Based on the provided query results, I can tell you that the following gamers are interested in the game "FIFA 21":\n\n* Patricia Martinez (Gamer ID: 10)\n* Laura Miller (Gamer ID: 8)\n* Jane Smith (Gamer ID: 2)\n* Robert Brown (Gamer ID: 3)\n* James Davis (Gamer ID: 9)\n* John Doe (Gamer ID: 1)\n* Emily Jones (Gamer ID: 6)\n\nNote that there is no additional information provided about their level of interest or any specific details about their involvement with the game.'}
```