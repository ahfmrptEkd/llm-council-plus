import gradio as gr
import asyncio
import os
import json
from typing import List, Dict, Any

# Ensure backend modules can be imported
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.council import (
    stage1_collect_responses,
    stage2_collect_rankings,
    stage3_synthesize_final,
    reset_token_stats
)
from backend.config import COUNCIL_MODELS, CHAIRMAN_MODEL, ROUTER_TYPE

async def ask_council(question: str, router_type: str, models_str: str, chairman_model: str, progress=gr.Progress()):
    """
    Ask the LLM Council a question with dynamic configuration.
    """
    try:
        # Initialize
        buffer = ""
        reset_token_stats()
        
        # Parse models
        custom_models = None
        if models_str and models_str.strip():
            custom_models = [m.strip() for m in models_str.split(",") if m.strip()]
        
        # Use defaults if empty
        active_models = custom_models if custom_models else COUNCIL_MODELS
        active_chairman = chairman_model if chairman_model else CHAIRMAN_MODEL
        active_router = router_type if router_type else ROUTER_TYPE
        
        models_display = ", ".join([m.split("/")[-1] for m in active_models])
        chairman_display = active_chairman.split("/")[-1]
        
        buffer += f"# 🏛️ LLM Council Session\n"
        buffer += f"**System Mode:** `{active_router.upper()}`\n"
        buffer += f"**Council Members:** {models_display}\n"
        buffer += f"**Chairman:** {chairman_display}\n\n---\n\n"
        yield buffer

        # Stage 1: Collect individual responses
        progress(0.1, desc="Stage 1: Council is deliberating...")
        buffer += "## 🟡 Stage 1: Collecting individual responses...\n\n"
        yield buffer

        # Call Stage 1 with dynamic config
        stage1_results, tool_outputs = await stage1_collect_responses(
            question, 
            models=active_models,
            router_type=active_router
        )

        if not stage1_results:
            buffer += "\n❌ **Error:** The council failed to generate any responses.\n"
            yield buffer
            return

        # Format Stage 1 results
        buffer += f"### ✅ Received {len(stage1_results)} responses:\n"
        for res in stage1_results:
            model_name = res["model"].split("/")[-1]
            if res.get('error'):
                buffer += f"- **{model_name}**: ⚠️ Failed ({res.get('error_message')})\n"
            else:
                preview = res["response"][:100].replace("\n", " ") + "..."
                buffer += f"- **{model_name}**: {preview}\n"
        
        # Show tool outputs if any
        if tool_outputs:
            buffer += f"\n**🔍 Research/Tools Used:**\n"
            for t in tool_outputs:
                buffer += f"- *{t['tool']}*: Found {len(t['result'])} chars of data\n"

        buffer += "\n---\n\n"
        yield buffer

        # Stage 2: Collect rankings
        progress(0.4, desc="Stage 2: Council is voting...")
        buffer += "## 🟡 Stage 2: Council members are ranking responses...\n\n"
        yield buffer

        # Call Stage 2 with dynamic config
        stage2_results, label_to_model = await stage2_collect_rankings(
            question, 
            stage1_results,
            models=active_models,
            router_type=active_router
        )

        # Format Stage 2 results
        buffer += "### ✅ Rankings Collected:\n"
        ranking_count = 0
        for res in stage2_results:
            model_name = res["model"].split("/")[-1]
            if not res.get('error'):
                ranking_count += 1
                buffer += f"- **{model_name}** has submitted their rankings.\n"
        
        if ranking_count == 0:
             buffer += "⚠️ No successful rankings received (skipping detailed scoring).\n"

        buffer += "\n---\n\n"
        yield buffer

        # Stage 3: Synthesize final answer
        progress(0.7, desc="Stage 3: Chairman is synthesizing...")
        buffer += "## 🟡 Stage 3: Chairman is synthesizing the final answer...\n\n"
        yield buffer

        # Call Stage 3 with dynamic config
        stage3_result = await stage3_synthesize_final(
            question, 
            stage1_results, 
            stage2_results,
            chairman=active_chairman,
            tool_outputs=tool_outputs,
            router_type=active_router
        )

        full_response = stage3_result.get("response", "")
        
        progress(1.0, desc="Session Adjourned")

        if stage3_result.get('error'):
             buffer += f"\n❌ **Error:** {full_response}\n"
             yield buffer
             return

        # Replace pending status with complete status
        final_buffer = buffer.replace(
            "## 🟡 Stage 3: Chairman is synthesizing the final answer...", 
            "## 🟢 Stage 3: Final Resolution"
        )
        
        # Add the final answer
        final_output = final_buffer + full_response
        yield final_output

    except Exception as e:
        import traceback
        traceback.print_exc()
        yield f"❌ **Critical Error during council session:** {str(e)}"


description = f"""
# LLM Council Plus 🏛️
An advanced AI council system where multiple LLMs discuss, rank, and synthesize answers to your questions.

### Current Configuration (Default)
- **Router Mode:** `{ROUTER_TYPE.upper()}`
- **Models:** {len(COUNCIL_MODELS)} Active (Chairman: {CHAIRMAN_MODEL.split('/')[-1]})
- **MCP Server:** Active

**How to use:**
1. Configure specific models in "Advanced Configuration" if needed.
2. Type your question below.
3. Wait for the council to deliberate (Stage 1 -> 2 -> 3).
"""

with gr.Blocks(theme=gr.themes.Soft(), title="LLM Council Plus") as demo:
    gr.Markdown(description)
    


    # User-specified defaults
    # OpenRouter needs full prefixes
    OPENROUTER_DEFAULTS = {
        "models": [
            "openai/gpt-5.1",
            "google/gemini-3-pro-preview",
            "anthropic/claude-opus-4.5",
            "x-ai/grok-4"
        ],
        "chairman": "google/gemini-3-pro-preview"
    }
    
    # Direct mode usually needs compatible IDs (Azure deployments often don't have prefixes)
    # Based on llm_manager.py, gemini/anthropic/grok prefixes are stripped automatically, 
    # but openai/ prefix might cause issues with Azure.
    DIRECT_DEFAULTS = {
        "models": [
            "gpt-5.1",
            "gemini-3-pro",
            "claude-opus-4.5",
            "grok-4"
        ],
        "chairman": "gemini-3-pro"
    }

    def update_config_defaults(router_choice):
        """Update model fields based on router type."""
        if router_choice == "openrouter":
            return (
                ", ".join(OPENROUTER_DEFAULTS["models"]), 
                OPENROUTER_DEFAULTS["chairman"]
            )
        else:
            return (
                ", ".join(DIRECT_DEFAULTS["models"]), 
                DIRECT_DEFAULTS["chairman"]
            )

    with gr.Accordion("Advanced Configuration", open=False):
        with gr.Row():
            router_type_input = gr.Radio(
                ["openrouter", "direct"], 
                label="Router Type", 
                value=ROUTER_TYPE,
                info="Select 'direct' for local Ollama/Azure/Google, 'openrouter' for cloud."
            )
            chairman_input = gr.Textbox(
                label="Chairman Model", 
                value=OPENROUTER_DEFAULTS["chairman"] if ROUTER_TYPE == "openrouter" else DIRECT_DEFAULTS["chairman"],
                info="Model ID for the final synthesizer"
            )
        
        models_input = gr.Textbox(
            label="Council Models", 
            value=",".join(OPENROUTER_DEFAULTS["models"] if ROUTER_TYPE == "openrouter" else DIRECT_DEFAULTS["models"]),
            info="Comma-separated list of model IDs"
        )
        
        # Event listener for router type change
        router_type_input.change(
            fn=update_config_defaults,
            inputs=[router_type_input],
            outputs=[models_input, chairman_input]
        )



    chatbot = gr.Markdown(height=600, label="Council Record")
    msg = gr.Textbox(lines=2, placeholder="Ask the council a question...", label="Question")
    
    with gr.Row():
        clear = gr.Button("Clear")
        submit = gr.Button("Submit", variant="primary")

    submit.click(
        ask_council, 
        inputs=[msg, router_type_input, models_input, chairman_input], 
        outputs=[chatbot]
    )
    msg.submit(
        ask_council, 
        inputs=[msg, router_type_input, models_input, chairman_input], 
        outputs=[chatbot]
    )
    clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    # Launch with mcp_server=True to expose as MCP
    # Use 0.0.0.0 for Docker/Spaces compatibility
    demo.launch(server_name="0.0.0.0", server_port=7860, mcp_server=True, show_error=True)
