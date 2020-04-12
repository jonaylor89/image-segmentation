
IMAGE="jonaylor/cmsc_project_2"

.PHONY: build
build: clean
	docker build -t $(IMAGE) .

.PHONY: run
run: build
	docker run -v $(HOME)/Repos/CMSC_630_Project_2/datasets:/app/datasets $(IMAGE)

.PHONY: clean
clean:
	rm -rf ./datasets/output