"""Cost Logger for tracking LLM token usage and costs."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import yaml



def _load_model_pricing() -> Dict[str, Dict[str, float]]:
    """Load model pricing from YAML config file.

    Flattens nested structure into single-level dict.

    Returns:
        Dict mapping model names to pricing info
    """
    config_path = Path(__file__).parent / "config" / "model_pricing.yaml"

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        pricing = {}
        for category, models in config.items():
            if isinstance(models, dict):
                pricing.update(models)

        return pricing
    except FileNotFoundError:
        print(f"Warning: Pricing config not found: {config_path}")
        return {}
    except Exception as e:
        print(f"Warning: Error loading pricing config: {e}")
        return {}


class CostLogger:
    """Logger for tracking LLM token usage and associated costs."""

    PRICING = _load_model_pricing()

    def __init__(self, log_dir: str = "logs"):
        """Initialize the cost logger.

        Args:
            log_dir: Directory to store log files
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.current_session = {
            "start_time": datetime.now().isoformat(),
            "requests": [],
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_cost": 0.0,
        }

    def log_request(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        prompt: Optional[str] = None,
        response: Optional[str] = None,
        metadata: Optional[Dict] = None,
        print_immediately: bool = False,
        langsmith_tracking: Optional[bool] = None,
    ) -> Dict:
        """Log a single LLM request.

        Args:
            model: Model name used
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            prompt: The prompt sent to the model (optional)
            response: The model's response (optional)
            metadata: Additional metadata (optional)
            print_immediately: Whether to print cost info immediately (default: False)
            langsmith_tracking: Whether Langsmith tracking is enabled for this request (optional)

        Returns:
            Dictionary with cost information
        """
        cost_info = self.calculate_cost(model, input_tokens, output_tokens)

        # Fallback to alias if cost is 0 and alias is provided (for local pricing lookup)
        if cost_info["total_cost"] == 0 and metadata and isinstance(metadata, dict) and "model_alias" in metadata:
            alias_cost_info = self.calculate_cost(metadata["model_alias"], input_tokens, output_tokens)
            if alias_cost_info["total_cost"] > 0:
                cost_info = alias_cost_info

        # Create request entry
        request_entry = {
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "cost": cost_info["total_cost"],
            "input_cost": cost_info["input_cost"],
            "output_cost": cost_info["output_cost"],
        }

        if langsmith_tracking is not None:
            request_entry["langsmith_tracking"] = langsmith_tracking
            request_entry["token_source"] = "api_response" if langsmith_tracking else "estimated"

        if metadata:
            request_entry["metadata"] = metadata

        self.current_session["requests"].append(request_entry)
        self.current_session["total_input_tokens"] += input_tokens
        self.current_session["total_output_tokens"] += output_tokens
        self.current_session["total_cost"] += cost_info["total_cost"]

        if print_immediately:
            self._print_cost_info(cost_info, input_tokens, output_tokens)

        return cost_info

    def calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> Dict:
        """Calculate cost using local pricing config.

        Args:
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Dictionary with cost breakdown
        """
        pricing = {}
        total_cost = 0.0
        input_cost = 0.0
        output_cost = 0.0

        # Fallback to Local Pricing
        if model in self.PRICING:
            pricing = self.PRICING[model]  # returns {input: x, output: y} per 1k
            
            # Ensure pricing is a dict (handle unexpected types)
            if not isinstance(pricing, dict):
                self._log_missing_price(model, f"Pricing config is not a dict: {type(pricing)}")
                pricing = {}

            # YAML prices are per 1000 tokens
            input_price_per_token = pricing.get("input", 0) / 1000.0
            output_price_per_token = pricing.get("output", 0) / 1000.0

            input_cost = input_tokens * input_price_per_token
            output_cost = output_tokens * output_price_per_token
            total_cost = input_cost + output_cost
        else:
            self._log_missing_price(model, "Cost calculated as 0 (Not in Local Config)")


        return {"input_cost": input_cost, "output_cost": output_cost, "total_cost": total_cost, "pricing": pricing}

    def _log_missing_price(self, model: str, error: str):
        """Log missing model pricing to a file."""
        log_file = self.log_dir / "missing_litellm_prices.log"
        timestamp = datetime.now().isoformat()
        try:
            with open(log_file, "a") as f:
                f.write(f"[{timestamp}] Model: {model} | Error: {error}\n")
        except Exception as e:
            print(f"Failed to log missing price: {e}")

    def _print_cost_info(self, cost_info: Dict, input_tokens: int, output_tokens: int):
        """Print cost information to console."""
        print(f"\n{'=' * 60}")
        print("Token Usage:")
        print(f"  Input tokens:  {input_tokens:,}")
        print(f"  Output tokens: {output_tokens:,}")
        print(f"  Total tokens:  {input_tokens + output_tokens:,}")
        print("\nCost Breakdown:")
        print(f"  Input cost:  ${cost_info['input_cost']:.6f}")
        print(f"  Output cost: ${cost_info['output_cost']:.6f}")
        print(f"  Total cost:  ${cost_info['total_cost']:.6f}")
        print(f"{'=' * 60}\n")

    def print_session_summary(self, show_details: bool = True):
        """Print summary of current session.

        Args:
            show_details: Whether to show individual request details (default: True)
        """
        print(f"\n{'=' * 60}")
        print("SESSION SUMMARY")
        print(f"{'=' * 60}")
        print(f"Start time: {self.current_session['start_time']}")
        print(f"Total requests: {len(self.current_session['requests'])}")

        langsmith_count = sum(1 for req in self.current_session["requests"] if req.get("langsmith_tracking", False))
        if langsmith_count > 0:
            print(f"Langsmith tracked requests: {langsmith_count}/{len(self.current_session['requests'])}")

        if show_details and self.current_session["requests"]:
            print(f"\n{'─' * 60}")
            print("Request Details:")
            print(f"{'─' * 60}")
            for i, req in enumerate(self.current_session["requests"], 1):
                stage = req.get("metadata", {}).get("stage", "unknown")
                print(f"\nRequest {i} ({stage}):")
                print(f"  Model: {req['model']}")
                print(f"  Input tokens:  {req['input_tokens']:,}")
                print(f"  Output tokens: {req['output_tokens']:,}")
                print(f"  Total tokens:  {req['total_tokens']:,}")
                print(f"  Cost: ${req['cost']:.6f}")

                if req.get("langsmith_tracking") is not None:
                    tracking_status = "✓ Langsmith" if req["langsmith_tracking"] else "✗ Langsmith"
                    token_source = req.get("token_source", "unknown")
                    print(f"  Tracking: {tracking_status} ({token_source})")

        print(f"\n{'─' * 60}")
        print("Total Token Usage:")
        print(f"  Input tokens:  {self.current_session['total_input_tokens']:,}")
        print(f"  Output tokens: {self.current_session['total_output_tokens']:,}")
        print(f"  Total tokens:  {self.current_session['total_input_tokens'] + self.current_session['total_output_tokens']:,}")
        print(f"\nTotal Cost: ${self.current_session['total_cost']:.6f}")
        print(f"{'=' * 60}\n")

        if langsmith_count > 0:
            print(f"💡 Note: {langsmith_count} request(s) were tracked by Langsmith.")
            print("   Compare results at: https://smith.langchain.com/")
            print()

    def save_session(self, filename: Optional[str] = None):
        """Save current session to a JSON file.

        Args:
            filename: Optional filename, defaults to timestamp-based name
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"cost_log_{timestamp}.json"

        filepath = self.log_dir / filename

        # Add end time
        self.current_session["end_time"] = datetime.now().isoformat()

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.current_session, f, indent=2, ensure_ascii=False)

        return filepath

    def get_session_stats(self) -> Dict:
        """Get current session statistics."""
        return {
            "total_requests": len(self.current_session["requests"]),
            "total_input_tokens": self.current_session["total_input_tokens"],
            "total_output_tokens": self.current_session["total_output_tokens"],
            "total_tokens": self.current_session["total_input_tokens"] + self.current_session["total_output_tokens"],
            "total_cost": self.current_session["total_cost"],
        }
