# Makefile
.PHONY: dev app train infer report lint test

PY=python

dev:
	$(PY) -m pip install -r requirements.txt

app:
	streamlit run app/main.py

train:
	$(PY) -c "from core.style.dataset import build_corpus; from core.style.space import BrandStyleSpace; from core.softprompt.trainer import SoftPromptTrainer, SoftPromptWeights; import pathlib; d=pathlib.Path('data/uploads'); chunks=build_corpus(str(d)); space=BrandStyleSpace.fit(chunks); pathlib.Path('data/brand/brand_style_space.json').write_text(space.to_json(), encoding='utf-8'); sp=SoftPromptTrainer(tokens=40, dim=256).fit(chunks); pathlib.Path('core/softprompt/weights').mkdir(parents=True, exist_ok=True); SoftPromptTrainer.save(sp, 'core/softprompt/weights/brand_softprompt.json'); print('OK')"

infer:
	$(PY) -c "from core.pipeline.rewriter import BrandVoiceRewriter, TaskSpec; from core.style.space import BrandStyleSpace; from core.softprompt.trainer import SoftPromptTrainer; import pathlib; draft='We help teams ship faster with a developer-first analytics layer.'; space=BrandStyleSpace.from_json(pathlib.Path('data/brand/brand_style_space.json').read_text(encoding='utf-8')); sp=SoftPromptTrainer.load('core/softprompt/weights/brand_softprompt.json'); pipe=BrandVoiceRewriter(space, sp); out=pipe.rewrite(draft, TaskSpec(tone='friendly')); print(out['best']['text'])"

report:
	$(PY) -c "print('Reports are generated from the Streamlit UI (tab 3/4).')"

lint:
	$(PY) -m pip install ruff && ruff check .

test:
	$(PY) -m pip install pytest && pytest -q
