.PHONY: setup train viz rewrite report clean

PY=python

setup:
	$(PY) -m pip install -r requirements.txt
	mkdir -p data reports figures adapters

# Example: make train FILES="data/corpus.docx data/notes.pdf" OUT=adapters/my_brand
train:
	@[ -n "$(FILES)" ] || (echo "FILES is empty. e.g. make train FILES='data/a.docx data/b.pdf' OUT=adapters/my_brand"; exit 1)
	@[ -n "$(OUT)" ] || (echo "OUT is empty. e.g. OUT=adapters/my_brand"; exit 1)
	$(PY) -m pipeline.train_gemma --files $(FILES) --out $(OUT) $(EXTRA)

# Example: make viz ADAPTER=adapters/my_brand
viz:
	@[ -n "$(ADAPTER)" ] || (echo "ADAPTER is empty. e.g. ADAPTER=adapters/my_brand"; exit 1)
	$(PY) -m pipeline.visualize_softprompt --adapter $(ADAPTER) --out_dir figures

# Example: make rewrite ADAPTER=adapters/my_brand TEXT="Your draft here"
rewrite:
	@[ -n "$(ADAPTER)" ] || (echo "ADAPTER is empty. e.g. ADAPTER=adapters/my_brand"; exit 1)
	@[ -n "$(TEXT)" ] || (echo "TEXT is empty. e.g. TEXT='We build tools…'"; exit 1)
	$(PY) -m pipeline.rewrite --adapter $(ADAPTER) --text "$(TEXT)"

# Example: make report ADAPTER=adapters/my_brand
report:
	@[ -n "$(ADAPTER)" ] || (echo "ADAPTER is empty. e.g. ADAPTER=adapters/my_brand"; exit 1)
	$(PY) -m pipeline.posttrain_report --adapter $(ADAPTER) --fig_dir figures --out reports

clean:
	rm -rf adapters/* figures/* reports/*
