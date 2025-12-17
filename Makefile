.PHONY: lint

# Fix all linting and formatting issues from CI
lint:
	@echo "Installing ruff..."
	@python -m pip install --upgrade pip > /dev/null 2>&1
	@pip install ruff==0.6.9 > /dev/null 2>&1
	@echo "✓ Ruff installed"
	@echo ""
	@echo "Fixing linting issues..."
	@ruff check --fix .
	@echo ""
	@echo "Fixing formatting..."
	@ruff format .
	@echo ""
	@echo "✓ All linting and formatting issues fixed!"
	@echo ""
	@echo "Verifying fixes..."
	@ruff check .
	@ruff format --check .
	@echo ""
	@echo "✓ All CI lint checks will pass!"
