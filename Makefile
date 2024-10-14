docker-build: ## make docker-build SERVICE=htrflow TAG=v0.1.0
	@docker build -t airiksarkivet/$(SERVICE):$(if $(TAG),$(TAG),latest) -f docker/$(SERVICE).dockerfile .

docker-tag: ## make docker-tag SERVICE=htrflow TAG=v0.1.0 REGISTRY=registry.ra.se:5002
	@docker tag airiksarkivet/$(SERVICE):$(if $(TAG),$(TAG),latest) $(REGISTRY)/airiksarkivet/$(SERVICE):$(if $(TAG),$(TAG),latest)

docker-push: ## make docker-push SERVICE=htrflow TAG=v0.1.0 REGISTRY=registry.ra.se:5002
	@docker push $(REGISTRY)/airiksarkivet/$(SERVICE):$(if $(TAG),$(TAG),latest)

docker-release: docker-build docker-tag docker-push ## make docker-release SERVICE=htrflow TAG=v0.1.0 REGISTRY=registry.ra.se:5002
	@echo "Docker image built, tagged, and pushed successfully!"
