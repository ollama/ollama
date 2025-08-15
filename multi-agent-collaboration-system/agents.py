from crewai import Agent, Task, Crew, Process
from langchain_community.llms import Ollama
from tools import SearxNGSearchTool, ComfyUITool

# Initialize the tool
search_tool = SearxNGSearchTool()

# Initialize the Ollama model
ollama_llm = Ollama(model="llama2")

# Define the researcher agent
researcher = Agent(
    role='Senior Research Analyst',
    goal='Uncover cutting-edge developments in AI and data science',
    backstory="""You are a Senior Research Analyst at a leading tech think tank.
    Your expertise lies in identifying emerging trends and technologies in AI and data science.
    You have a knack for dissecting complex data and presenting actionable insights.""",
    verbose=True,
    allow_delegation=False,
    tools=[search_tool],
    llm=ollama_llm
)

# Define the writer agent
writer = Agent(
    role='Tech Content Strategist',
    goal='Craft compelling content on tech advancements',
    backstory="""You are a renowned Tech Content Strategist, known for your insightful and engaging articles.
    You transform complex technical concepts into accessible and compelling narratives.""",
    verbose=True,
    allow_delegation=True,
    llm=ollama_llm
)

# Define the research task
research_task = Task(
    description="""Conduct a comprehensive analysis of the latest advancements in AI in 2024.
    Identify key trends, breakthrough technologies, and potential industry impacts.""",
    expected_output="A full analysis report with detailed sections on key trends, technologies, and impacts.",
    agent=researcher
)

# Define the writing task
write_task = Task(
    description="""Using the research findings, write an insightful blog post about the top 5 AI advancements of 2024.
    Your post should be informative, engaging, and accessible to a tech-savvy audience.
    Make it sound cool, avoid complex words so it doesn't sound like AI.""",
    expected_output="A 1500-word blog post in markdown format, with a catchy title and clear sections.",
    agent=writer
)

# Create the research crew
research_crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    process=Process.sequential,
    verbose=True
)

# Define the illustrator agent
illustrator = Agent(
    role='Image Illustrator',
    goal='Create compelling images based on text prompts',
    backstory="""You are a skilled illustrator who can create stunning images from text prompts.
    You are an expert in using ComfyUI to generate images.""",
    verbose=True,
    allow_delegation=False,
    tools=[ComfyUITool()],
    llm=ollama_llm
)

# Define the image generation task
image_task = Task(
    description="""Generate an image based on the following prompt: A futuristic cityscape with flying cars.""",
    expected_output="A URL to the generated image.",
    agent=illustrator
)

# Create the image generation crew
image_crew = Crew(
    agents=[illustrator],
    tasks=[image_task],
    process=Process.sequential,
    verbose=True
)


if __name__ == '__main__':
    # Kick off the research crew's work
    # result = research_crew.kickoff()
    # print("######################")
    # print(result)

    # Kick off the image crew's work
    image_result = image_crew.kickoff()
    print("######################")
    print(image_result)
