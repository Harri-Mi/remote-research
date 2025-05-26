from dotenv import load_dotenv
import ollama # Replaced anthropic with ollama
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client  # Add this import at the top

from contextlib import AsyncExitStack
import json
import asyncio
import nest_asyncio

nest_asyncio.apply()

load_dotenv()

class MCP_ChatBot:
    def __init__(self):
        self.exit_stack = AsyncExitStack()
        self.ollama = ollama.AsyncClient() # Replaced Anthropic with ollama.AsyncClient
        # Tools list required for Anthropic API
        self.available_tools = []
        # Prompts list for quick display 
        self.available_prompts = []
        # Sessions dict maps tool/prompt names or resource URIs to MCP client sessions
        self.sessions = {}

    async def connect_to_server(self, server_name, server_config):
        try:
            # --- Support both stdio and remote (SSE) servers ---
            if "url" in server_config:
                # Connect via SSE/HTTP
                sse_transport = await self.exit_stack.enter_async_context(
                    sse_client(server_config["url"])
                )
                read, write = sse_transport
            else:
                # Connect via stdio (local process)
                server_params = StdioServerParameters(**server_config)
                stdio_transport = await self.exit_stack.enter_async_context(
                    stdio_client(server_params)
                )
                read, write = stdio_transport

            session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            await session.initialize()
            
            
            try:
                # List available tools
                response = await session.list_tools()
                for tool in response.tools:
                    self.sessions[tool.name] = session
                    self.available_tools.append({
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.inputSchema
                    })
            
                # List available prompts
                prompts_response = await session.list_prompts()
                if prompts_response and prompts_response.prompts:
                    for prompt in prompts_response.prompts:
                        self.sessions[prompt.name] = session
                        self.available_prompts.append({
                            "name": prompt.name,
                            "description": prompt.description,
                            "arguments": prompt.arguments
                        })
                # List available resources
                resources_response = await session.list_resources()
                if resources_response and resources_response.resources:
                    for resource in resources_response.resources:
                        resource_uri = str(resource.uri)
                        self.sessions[resource_uri] = session
            
            except Exception as e:
                print(f"Error {e}")
                
        except Exception as e:
            print(f"Error connecting to {server_name}: {e}")

    async def connect_to_servers(self):
        try:
            with open("server_config.json", "r") as file:
                data = json.load(file)
            servers = data.get("mcpServers", {})
            for server_name, server_config in servers.items():
                await self.connect_to_server(server_name, server_config)
        except Exception as e:
            print(f"Error loading server config: {e}")
            raise
    
    async def process_query(self, query):
        messages = [{'role':'user', 'content':query}]
        
        # Construct a system prompt that includes tool descriptions
        # This is a common way to make LLMs aware of available tools
        system_prompt_parts = ["You have the following tools available:"]
        for tool in self.available_tools:
            system_prompt_parts.append(f"- Name: {tool['name']}")
            system_prompt_parts.append(f"  Description: {tool['description']}")
            system_prompt_parts.append(f"  Input Schema: {json.dumps(tool['input_schema'])}")
        system_prompt_parts.append("If you need to use a tool, respond with a JSON object with two keys: 'tool_name' and 'tool_input'. For example: {\"tool_name\": \"tool_name\", \"tool_input\": {\"arg1\": \"value1\"}}.")
        system_message = {"role": "system", "content": "\n".join(system_prompt_parts)}
        
        # Prepend system message to messages list if not already present or if it's different
        # This is a simplified check; a more robust solution might involve checking content
        if not messages or messages[0]['role'] != 'system':
            current_messages = [system_message] + messages
        else:
            current_messages = messages # Assume system prompt is already there and up-to-date for subsequent turns

        while True:
            # response = await self.ollama.chat(
            #     model='llama2', # Changed model name
            #     messages=current_messages
            # )
            response = await self.ollama.chat(
                model='gemma:2b', # Changed model name to gemma:2b
                messages=current_messages
            )
            
            assistant_message = response['message']
            assistant_content_text = assistant_message['content']
            print(assistant_content_text) # Print model's text response

            messages.append({'role': 'assistant', 'content': assistant_content_text})
            current_messages.append({'role': 'assistant', 'content': assistant_content_text})


            has_tool_use = False
            try:
                # Attempt to parse the assistant's response as JSON for tool use
                # This is a common convention when direct tool support isn't available
                tool_call_info = json.loads(assistant_content_text)
                if isinstance(tool_call_info, dict) and 'tool_name' in tool_call_info and 'tool_input' in tool_call_info:
                    tool_name = tool_call_info['tool_name']
                    tool_input = tool_call_info['tool_input']

                    if tool_name in [t['name'] for t in self.available_tools]:
                        has_tool_use = True
                        print(f"Attempting to call tool: {tool_name} with input: {tool_input}")

                        session = self.sessions.get(tool_name)
                        if not session:
                            print(f"Tool '{tool_name}' not found in active sessions.")
                            # Add a message to current_messages indicating tool not found
                            tool_result_message = {
                                'role': 'user', # Or 'tool' if Ollama supports it, user for now
                                'content': f"Error: Tool '{tool_name}' not found or session unavailable."
                            }
                            messages.append(tool_result_message)
                            current_messages.append(tool_result_message)
                            break 

                        result = await session.call_tool(tool_name, arguments=tool_input)
                        
                        # Construct the tool result message to send back to the model
                        # The format might need adjustment based on how Ollama expects tool results
                        tool_result_content = result.content if hasattr(result, 'content') else str(result)
                        tool_result_message = {
                            'role': 'user', # Simulating a user message that provides tool results
                            'content': f"Tool {tool_name} execution result: {tool_result_content}"
                        }
                        messages.append(tool_result_message)
                        current_messages.append(tool_result_message)
                    else:
                        # Model generated JSON but not a valid tool or format
                        pass # Continue as a normal text response
            except json.JSONDecodeError:
                # Assistant content is not JSON, so it's a regular text response
                pass
            except Exception as e:
                print(f"Error processing tool call: {e}")
                # Add error message to current_messages
                error_message = {
                    'role': 'user',
                    'content': f"Error processing tool call: {e}"
                }
                messages.append(error_message)
                current_messages.append(error_message)
                break # Exit loop on other errors

            if not has_tool_use:
                break

    async def get_resource(self, resource_uri):
        session = self.sessions.get(resource_uri)
        
        # Fallback for papers URIs - try any papers resource session
        if not session and resource_uri.startswith("papers://"):
            for uri, sess in self.sessions.items():
                if uri.startswith("papers://"):
                    session = sess
                    break
            
        if not session:
            print(f"Resource '{resource_uri}' not found.")
            return
        
        try:
            result = await session.read_resource(uri=resource_uri)
            if result and result.contents:
                print(f"\nResource: {resource_uri}")
                print("Content:")
                print(result.contents[0].text)
            else:
                print("No content available.")
        except Exception as e:
            print(f"Error: {e}")
    
    async def list_prompts(self):
        """List all available prompts."""
        if not self.available_prompts:
            print("No prompts available.")
            return
        
        print("\nAvailable prompts:")
        for prompt in self.available_prompts:
            print(f"- {prompt['name']}: {prompt['description']}")
            if prompt['arguments']:
                print(f"  Arguments:")
                for arg in prompt['arguments']:
                    arg_name = arg.name if hasattr(arg, 'name') else arg.get('name', '')
                    print(f"    - {arg_name}")
    
    async def execute_prompt(self, prompt_name, args):
        """Execute a prompt with the given arguments."""
        session = self.sessions.get(prompt_name)
        if not session:
            print(f"Prompt '{prompt_name}' not found.")
            return
        
        try:
            result = await session.get_prompt(prompt_name, arguments=args)
            if result and result.messages:
                prompt_content = result.messages[0].content
                
                # Extract text from content (handles different formats)
                if isinstance(prompt_content, str):
                    text = prompt_content
                elif hasattr(prompt_content, 'text'):
                    text = prompt_content.text
                else:
                    # Handle list of content items
                    text = " ".join(item.text if hasattr(item, 'text') else str(item) 
                                  for item in prompt_content)
                
                print(f"\nExecuting prompt '{prompt_name}'...")
                await self.process_query(text)
        except Exception as e:
            print(f"Error: {e}")
    
    async def chat_loop(self):
        print("\nMCP Chatbot Started!")
        print("Type your queries or 'quit' to exit.")
        print("Use @folders to see available topics")
        print("Use @<topic> to search papers in that topic")
        print("Use /prompts to list available prompts")
        print("Use /prompt <name> <arg1=value1> to execute a prompt")
        
        while True:
            try:
                query = input("\nQuery: ").strip()
                if not query:
                    continue
        
                if query.lower() == 'quit':
                    break
                
                # Check for @resource syntax first
                if query.startswith('@'):
                    # Remove @ sign  
                    topic = query[1:]
                    if topic == "folders":
                        resource_uri = "papers://folders"
                    else:
                        resource_uri = f"papers://{topic}"
                    await self.get_resource(resource_uri)
                    continue
                
                # Check for /command syntax
                if query.startswith('/'):
                    parts = query.split()
                    command = parts[0].lower()
                    
                    if command == '/prompts':
                        await self.list_prompts()
                    elif command == '/prompt':
                        if len(parts) < 2:
                            print("Usage: /prompt <name> <arg1=value1> <arg2=value2>")
                            continue
                        
                        prompt_name = parts[1]
                        args = {}
                        
                        # Parse arguments
                        for arg in parts[2:]:
                            if '=' in arg:
                                key, value = arg.split('=', 1)
                                args[key] = value
                        
                        await self.execute_prompt(prompt_name, args)
                    else:
                        print(f"Unknown command: {command}")
                    continue
                
                await self.process_query(query)
                    
            except Exception as e:
                print(f"\nError: {str(e)}")
    
    async def cleanup(self):
        await self.exit_stack.aclose()


async def main():
    chatbot = MCP_ChatBot()
    try:
        await chatbot.connect_to_servers()
        await chatbot.chat_loop()
    finally:
        await chatbot.cleanup()


if __name__ == "__main__":
    asyncio.run(main())