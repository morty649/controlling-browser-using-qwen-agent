fine-tune:
	uv run modal run -m src.browser_control_using_grpo_qwen.fine_tuning --config-file-name $(config)

evaluation:
	uv run python -m src.browser_control_using_grpo_qwen.evaluate