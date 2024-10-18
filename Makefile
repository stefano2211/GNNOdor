.PHONY: tests docs

deps: 
	@echo "Initializing Git..."
	git init
	
	@echo "Installing dependencies..."
	pip install -r requirements-dev.txt
	pre-commit install
	

docs:
	@echo Save documentation to docs... 
	pdoc app.api -o docs --force
	@echo View API documentation... 
	pdoc app.api --http localhost:8080	
