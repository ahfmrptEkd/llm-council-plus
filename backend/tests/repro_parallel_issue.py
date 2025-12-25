import asyncio
import logging
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.llm.llm_manager import LLMManager
import litellm

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger("repro_parallel")

async def test_parallel_calls(apply_fix: bool = False, iterations: int = 5):
    if apply_fix:
        logger.info(f"=== Applying LiteLLM Fix (cache=None, drop_params=True) - Running {iterations} iterations ===")
        litellm.cache = None
        litellm.drop_params = True
    else:
        logger.info(f"=== Running WITHOUT Fix (Default LiteLLM behavior) - Running {iterations} iterations ===")

    manager = LLMManager()
    total_success = 0
    total_calls = 0
    
    # Models to test (Increased to 5 concurrent calls to match council max size)
    models = ["gpt-5", "gpt-5.1", "gpt-5", "gpt-5.1", "gpt-5"]
    prompt = "Hello, are you working? Respond with 'YES' and your model name."
    
    for i in range(iterations):
        logger.info(f"--- Iteration {i+1}/{iterations} ---")
        
        async def call_model(model_name: str, idx: int):
            try:
                # We need to manually get LLM instance because LLMManager handles config
                llm = manager.get_llm(model_name)
                result = await manager.invoke_with_tracking(llm, prompt, model_name)
                content = result.get("response_text", "")
                logger.info(f"Iter {i+1} [{idx}] {model_name} SUCCESS")
                return True
            except Exception as e:
                logger.error(f"Iter {i+1} [{idx}] {model_name} FAILED: {type(e).__name__}: {e}")
                return False

        # Execute in parallel
        tasks = [call_model(model, j) for j, model in enumerate(models)]
        results = await asyncio.gather(*tasks)
        
        total_success += sum(1 for r in results if r)
        total_calls += len(models)
        
        # Short sleep between iterations to avoid overwhelming API if too fast
        await asyncio.sleep(1)
    
    logger.info(f"Final Count: {total_success}/{total_calls} succeeded.")
    return total_success == total_calls

if __name__ == "__main__":
    # Run multiple iterations
    async def main():
        print("\n" + "="*50)
        print("TEST: STRESS TESTING PARALLEL CALLS (WITH FIX APPLIED)")
        # We've already applied the fix in shared/llm/llm_manager.py, 
        # but let's be explicit here.
        await test_parallel_calls(apply_fix=True, iterations=5)
    
    asyncio.run(main())
