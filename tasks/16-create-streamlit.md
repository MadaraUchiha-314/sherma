- Create a streamlit which allows a user design their own agent or multi-agent system with tools, skills, prompts
- Let's call this Agent Designer
- User can provide a natural langague prompt (through chat) and a fully agentic system with agent. prompts, tools, skills are created for them
- User can chat further to refine the agent that's created
- Once the full agent system is created, the user will be able to save and download the full agent and all the files created

- Model the Agent Designer itself as an agent (declarative agent)
    - Create the prompts, tools, skills requried for this Agent Designer Agent to work

- The declarative agent YAML and other files for tools and skills should be visible on the side panel when the user is chatting with the Agent Designer and iterating over their agent

- For the LLM call, provide a dropdown to choose the LLM provider:
    - openai
    - anthropic
- Add a text box to accept the API Key for either of the providers
- Create the httpx async client with the API Key
    - Look at examples/declarative_weather_agent/main.py fo reference


- Once the agent is created, the user should have an option to chat with that agent
    - Create a separate tab for this where all the agents created by the user are listed


- This streamlit should be hostable at https://streamlit.io/
- the api keys provided in streamlit should always be kept local and never be sent to any service
- Make the chat input and output multi-modal
    - Allow upload of images