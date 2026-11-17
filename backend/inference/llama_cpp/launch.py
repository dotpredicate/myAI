LAUNCH_ARGS: tuple[str, ...] = ("-ngl", "99", "-fa", "on")

# --fit off keeps the model's native context instead of shrinking it to fit;
# -np 1 gives that context to one sequence. llama-bench has no --fit.
SERVER_ARGS: tuple[str, ...] = LAUNCH_ARGS + ("--fit", "off", "-np", "1")
