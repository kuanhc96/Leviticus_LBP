start:
	docker build -t test_fastapi .
	docker run -d -p 8000:8000 test_fastapi	