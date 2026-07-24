.PHONY: demo run chat markup test

PID_A ?=
PID_B ?=

demo:   ; uv run delta-chat demo   --pid-a $(PID_A) --pid-b $(PID_B)
run:    ; uv run delta-chat run    --pid-a $(PID_A) --pid-b $(PID_B)
chat:   ; uv run delta-chat chat   --pid-a $(PID_A) --pid-b $(PID_B)
markup: ; uv run delta-chat markup --pid-a $(PID_A) --pid-b $(PID_B) --out markup.pdf

test:   ; uv run pytest
