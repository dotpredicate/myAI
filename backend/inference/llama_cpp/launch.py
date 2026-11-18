LAUNCH_ARGS: tuple[str, ...] = ("-ngl", "99", "-fa", "on")

# Without --ctx-size, llama.cpp reads n_ctx_train from the GGUF.
# --fit off fails at startup instead of shrinking that window.
# -np 1 gives the window to one request, so concurrent generations queue.
# llama-bench has no --fit.
SERVER_ARGS: tuple[str, ...] = LAUNCH_ARGS + ("--fit", "off", "-np", "1")
