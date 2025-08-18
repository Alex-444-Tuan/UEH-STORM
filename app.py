import os # module to interact with the operating system

import deepl
from dotenv import load_dotenv # module to load environment variables from a .env file
from langchain_groq import ChatGroq
import chainlit as cl
from utils.stormv2 import run_storm_pipeline
from langchain.callbacks.tracers import LangChainTracer
import asyncio
from langsmith import Client
import json
import re
from typing import List, Tuple
from langchain.agents import initialize_agent, Tool



load_dotenv()

#set up google translate

auth = os.getenv("DEEPL_API_KEY")

# dùng để chia table content dựa trên # ở trong file storm_gen_article.txt
def parse_all_markdown_sections(text: str) -> List[Tuple[str, str]]:
    pattern = r"(#+) (.+?)\n(.*?)(?=\n#+ |\Z)"
    matches = re.findall(pattern, text, flags=re.DOTALL)

    if not matches:
        return [("Toàn bộ nội dung", text.strip())]

    sections = []
    for hashes, raw_title, content in matches:
        level = len(hashes)
        prefix = "-" * level  # dùng - theo số cấp
        display_title = f"{prefix} {raw_title.strip()}"
        full_content = f"{hashes} {raw_title.strip()}\n\n{content.strip()}"
        sections.append((display_title, full_content))

    return sections




# client = Client(
#   api_key= os.getenv("LANGSMITH_API_KEY"), 
#   api_url="https://api.smith.langchain.com",  
# ) 
# # đây là optional có thể lượt bỏ bớt

@cl.on_chat_start
async def on_chat_start():
#DeepL
    translator = deepl.DeepLClient(auth_key=auth)
    cl.user_session.set("translator", translator)


@cl.on_message
async def handle_message(msgt: cl.Message):




# xử lí pipeline song ngữ và STORM
#google translate
    # msg = cl.user_session.get("client").translate(msgt.content, target_language=cl.user_session.get("output_language"))
# DeepL
    msg = cl.user_session.get("translator").translate_text(msgt.content, target_lang="EN-US", preserve_formatting=True)
    await cl.Message(content='Preparing the STORM Wiki pipeline...').send()
#google translate
    # topic = msg['translatedText']
# DeepL
    topic = msg.text 
    await cl.Message(content=f"Topic received: `{topic}`\nRunning the pipeline...").send()


    try:
        await cl.sleep(0.3)
        # chạy Pipeline Storm
        # asyncio dùng để chạy pipeline ở một block riêng biệt
        # để không làm gián đoạn quá trình xử lý của Chainlit
        await asyncio.to_thread(run_storm_pipeline,
            topic=topic,
            retriever="tavily",
            output_dir="./data",
            do_research=True,
            do_generate_outline=True,
            do_generate_article=True,
            do_polish_article=True
        )

        await asyncio.sleep(0.5)

        # Kiểm tra xem file đã được tạo ra chưa
        result_path = f"./data/{topic.replace(' ', '_').replace('/', '_')}/storm_gen_article.txt"
        if os.path.exists(result_path):
            with open(result_path, "r") as f:
                article = f.read()
#google translate
                # article_translated = cl.user_session.get("client").translate(article, target_language=cl.user_session.get("input_language"))['translatedText']
#DeepL
                article_translated = cl.user_session.get("translator").translate_text(article, target_lang="VI", preserve_formatting=True, tag_handling="xml").text
            await cl.Message(content=f"✅ Article generated:\n\n{article_translated}").send()
        else:
            await cl.Message(content="⚠️ Article file not found!").send()

        sections = parse_all_markdown_sections(article_translated)

        cl.user_session.set("sections", sections)  # Store sections in user session
        cl.user_session.set("topic", topic)  # Store topic in user session  

        toc = cl.TaskList(title=f"Table of Contents for {topic}", tasks=[cl.Task(title=title, status="done") for title, _ in sections])

        await toc.send()

        url_info_path = f"./data/{topic.replace(' ', '_').replace('/', '_')}/url_to_info.json"

        with open(url_info_path, "r") as f:
            full_info = json.load(f)

        # kiểm tra file link url 
        url_map = full_info.get("url_to_unified_index", {})

        if not url_map:
            await cl.Message(content="⚠️ `url_to_unified_index` not found or empty in `url_to_info.json`.").send()
        else:
            # Sort by index (value)
            sorted_urls = sorted(url_map.items(), key=lambda x: x[1])


            # lập table content
            markdown_links = "\n".join(
                f"{i+1}. [{url}]({url})" for i, (url, _) in enumerate(sorted_urls)
            )

            await cl.Message(content="🔗 **Sources used in article (from `url_to_info.json`)**:\n" + markdown_links).send()

    except Exception as e:
        await cl.Message(content=f" Error running STORM pipeline:\n```\n{str(e)}\n```").send()

    