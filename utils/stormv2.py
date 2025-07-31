import os

from argparse import ArgumentParser
from knowledge_storm import (
    STORMWikiRunnerArguments,
    STORMWikiRunner,
    STORMWikiLMConfigs,
)
from knowledge_storm.lm import OpenAIModel, AzureOpenAIModel
from knowledge_storm.rm import (
    YouRM,
    BingSearch,
    BraveRM,
    SerperRM,
    DuckDuckGoSearchRM,
    TavilySearchRM,
    SearXNG,
    AzureAISearch,
)
from knowledge_storm.utils import load_api_key

def run_storm_pipeline(
    topic: str,
    retriever: str = "you",
    output_dir: str = "./results/gpt",
    max_thread_num: int = 3,
    do_research: bool = True,
    do_generate_outline: bool = True,
    do_generate_article: bool = True,
    do_polish_article: bool = False,
    max_conv_turn: int = 3,
    max_perspective: int = 3,
    search_top_k: int = 3,
):
    load_api_key(toml_file_path="secrets.toml")
    lm_configs = STORMWikiLMConfigs()
    openai_kwargs = {
        "api_key": os.getenv("OPENAI_API_KEY"),
        "temperature": 1.0,
        "top_p": 0.9,
    }

    ModelClass = (
        OpenAIModel if os.getenv("OPENAI_API_TYPE") == "openai" else AzureOpenAIModel
    )
    # If you are using Azure service, make sure the model name matches your own deployed model name.
    # The default name here is only used for demonstration and may not match your case.
    gpt_35_model_name = (
        "gpt-3.5-turbo" if os.getenv("OPENAI_API_TYPE") == "openai" else "gpt-35-turbo"
    )
    gpt_4_model_name = "gpt-4o"
    if os.getenv("OPENAI_API_TYPE") == "azure":
        openai_kwargs["api_base"] = os.getenv("AZURE_API_BASE")
        openai_kwargs["api_version"] = os.getenv("AZURE_API_VERSION")

    # STORM is a LM system so different components can be powered by different models.
    # For a good balance between cost and quality, you can choose a cheaper/faster model for conv_simulator_lm
    # which is used to split queries, synthesize answers in the conversation. We recommend using stronger models
    # for outline_gen_lm which is responsible for organizing the collected information, and article_gen_lm
    # which is responsible for generating sections with citations.
    conv_simulator_lm = ModelClass(
        model=gpt_35_model_name, max_tokens=500, **openai_kwargs
    )
    question_asker_lm = ModelClass(
        model=gpt_35_model_name, max_tokens=500, **openai_kwargs
    )
    outline_gen_lm = ModelClass(model=gpt_4_model_name, max_tokens=400, **openai_kwargs)
    article_gen_lm = ModelClass(model=gpt_4_model_name, max_tokens=700, **openai_kwargs)
    article_polish_lm = ModelClass(
        model=gpt_4_model_name, max_tokens=4000, **openai_kwargs
    )

    lm_configs.set_conv_simulator_lm(conv_simulator_lm)
    lm_configs.set_question_asker_lm(question_asker_lm)
    lm_configs.set_outline_gen_lm(outline_gen_lm)
    lm_configs.set_article_gen_lm(article_gen_lm)
    lm_configs.set_article_polish_lm(article_polish_lm)

    engine_args = STORMWikiRunnerArguments(
        output_dir=output_dir,
        max_conv_turn=max_conv_turn,
        max_perspective=max_perspective,
        search_top_k=search_top_k,
        max_thread_num=max_thread_num,
    )

    match retriever:
        case "bing":
            rm = BingSearch(
                bing_search_api=os.getenv("BING_SEARCH_API_KEY"),
                k=engine_args.search_top_k,
            )
        case "you":
            rm = YouRM(ydc_api_key=os.getenv("YDC_API_KEY"), k=engine_args.search_top_k)
        case "brave":
            rm = BraveRM(
                brave_search_api_key=os.getenv("BRAVE_API_KEY"),
                k=engine_args.search_top_k,
            )
        case "duckduckgo":
            rm = DuckDuckGoSearchRM(
                k=engine_args.search_top_k, safe_search="On", region="us-en"
            )
        case "serper":
            rm = SerperRM(
                serper_search_api_key=os.getenv("SERPER_API_KEY"),
                query_params={"autocorrect": True, "num": 10, "page": 1},
            )
        case "tavily":
            rm = TavilySearchRM(
                tavily_search_api_key=os.getenv("TAVILY_API_KEY"),
                k=engine_args.search_top_k,
                include_raw_content=True,
            )
        case "searxng":
            rm = SearXNG(
                searxng_api_key=os.getenv("SEARXNG_API_KEY"), k=engine_args.search_top_k
            )
        case "azure_ai_search":
            rm = AzureAISearch(
                azure_ai_search_api_key=os.getenv("AZURE_AI_SEARCH_API_KEY"),
                k=engine_args.search_top_k,
            )
        case _:
            raise ValueError(
                f'Invalid retriever: {args.retriever}. Choose either "bing", "you", "brave", "duckduckgo", "serper", "tavily", "searxng", or "azure_ai_search"'
            )

    runner = STORMWikiRunner(engine_args, lm_configs, rm)

    runner.run(
        topic=topic,
        do_research=do_research,
        do_generate_outline=do_generate_outline,
        do_generate_article=do_generate_article,
        do_polish_article=do_polish_article,
    )
    runner.post_run()
    runner.summary()
    return runner