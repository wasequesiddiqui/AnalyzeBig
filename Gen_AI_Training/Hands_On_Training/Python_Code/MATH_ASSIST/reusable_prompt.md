# Task: Generate an Interactive HTML Explanation for [SYSTEM NAME]

You are an expert technical writer and front-end developer. Given the following technical system description, create a **single, self-contained HTML file** that explains the system in an interactive, lucid, and visually engaging way.

## System Details

- **System Name:** [e.g., YouTube Recursive AI Agent]
- **Core Technologies:** [e.g., LangChain, OpenAI, Python, yt-dlp]
- **Key Features / Tools:** [list main functions, e.g., extract_video_id, fetch_transcript, search_youtube]
- **Architecture Pattern:** [e.g., Recursive tool-calling, universal chain, parallel execution]
- **Sample User Query:** [e.g., "Summarize this YouTube video"]
- **Attached Diagram (optional):** [file name if any, or describe the visual]

## Requirements for the HTML

1. **Visual Style**  
   - Modern, clean, responsive (Tailwind CSS or similar).  
   - Use gradients, cards, icons (Font Awesome), and a dark/light balanced palette.  
   - Must be readable and engaging.

2. **Content Sections** (adapt as needed)  
   - **Hero section** with title, tagline, and tech badges.  
   - **Tool Overview** – cards describing each tool/function.  
   - **Architecture Diagram** – use Mermaid.js to illustrate the flow (e.g., user query → entry point → recursive loop → final answer).  
   - **Why This Pattern Works** – explain the core design principle (e.g., `universal_chain`).  
   - **Recursive Logic Deep Dive** – show pseudo-code or actual code snippets with explanations.  
   - **Interactive Simulation** – a step-by-step mock of the agent’s tool-calling flow (user can click to run a demo).  
   - **Code Reference** – collapsible block with the essential implementation.  
   - **Key Takeaways** – 3–4 bullet points.  
   - **Footer** with attribution.

3. **Attached Image**  
   - If an image file (e.g., `image.png`) is referenced, embed it with `<img src="image.png">` and add a fallback/error message if missing.

4. **Interactivity**  
   - A simulation log that shows the sequence of tool calls, AI messages, and final answer.  
   - Buttons to “Run Simulation” and “Reset Log”.  
   - The simulation should mirror the recursive logic without requiring real API calls (mock data).

5. **Code Representation**  
   - Use syntax-highlighted blocks (dark background, monospace).  
   - Keep code snippets accurate to the actual implementation.

6. **Clarity & Pedagogy**  
   - Use icons, color-coded sections, and simple language.  
   - Avoid jargon overload – explain each concept.  
   - Ensure the HTML is fully self-contained (CSS/JS inline or via CDN).

## Additional Context from the Original Conversation

The original conversation produced an HTML for a **YouTube Recursive AI Agent** with these specifics:  
- Tools: extract_video_id, fetch_transcript, search_youtube, get_full_metadata, get_thumbnails.  
- Recursion: `_recursive_chain` + `process_tool_calls`.  
- Universal chain entry point: `universal_chain = RunnableLambda(...)`.  
- Simulation: demonstrates the LLM calling tools in sequence and synthesizing a final answer.

Your generated HTML should follow the same template **but for my system**.

## Output Format

Return **only the complete HTML code** inside a single markdown code block. Do not include extra explanations.