export const RAGTemplate = (context: string, query: string) => {
	let template = `Use the following context as your learned knowledge, inside <context></context> XML tags.
	<context>
	  [context]
	</context>
	
	When answer to user:
	- If you don't know, just say that you don't know.
	- If you don't know when you are not sure, ask for clarification.
	Avoid mentioning that you obtained the information from the context.
	And answer according to the language of the user's question.
			
	Given the context information, answer the query.
	Query: [query]`;

	template = template.replace(/\[context\]/g, context);
	template = template.replace(/\[query\]/g, query);

	return template;
};
