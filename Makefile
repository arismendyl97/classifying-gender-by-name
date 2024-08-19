# Makefile for Name-Based Gender Classification Project

# Variables
PROJECT_DIR := $(shell pwd)
CONDA_ENV := mlopsproject
DOCKER_COMPOSE_FILE := docker-compose.yml
CLOUDFORMATION_TEMPLATE := cloudformation/aws_ec2_cloud-docker-app.yaml
STACK_NAME := MyDockerAppStack

# Default target
.PHONY: all
all: prepare_data train_model

# Create and activate conda environment
.PHONY: env
env:
	@echo "Creating and activating conda environment..."
	conda env create -f environment.yml
	conda activate $(CONDA_ENV)

# Prepare the dataset
.PHONY: prepare_data
prepare_data:
	@echo "Preparing dataset..."
	cd data_cleaning && python prepare_dataset.py

# Train the model
.PHONY: train_model
train_model:
	@echo "Training the model..."
	cd training && python training_model.py

# Register the best model with MLflow
.PHONY: register_model
register_model:
	@echo "Registering the best model..."
	cd training && python best_model.py

# Build and run the Docker containers
.PHONY: docker_up
docker_up:
	@echo "Building and running Docker containers..."
	docker-compose up --build

# Stop and remove Docker containers
.PHONY: docker_down
docker_down:
	@echo "Stopping and removing Docker containers..."
	docker-compose down

# Deploy the application to AWS using CloudFormation
.PHONY: deploy_cloud
deploy_cloud:
	@echo "Deploying application to AWS..."
	aws cloudformation create-stack \
	  --stack-name $(STACK_NAME) \
	  --template-body file://$(PROJECT_DIR)/$(CLOUDFORMATION_TEMPLATE) \
	  --capabilities CAPABILITY_IAM

# Delete the AWS CloudFormation stack
.PHONY: delete_cloud
delete_cloud:
	@echo "Deleting CloudFormation stack..."
	aws cloudformation delete-stack --stack-name $(STACK_NAME)

# Display help
.PHONY: help
help:
	@echo "Makefile for Name-Based Gender Classification Project"
	@echo "Usage:"
	@echo "  make all              - Prepare data and train the model"
	@echo "  make env              - Create and activate the conda environment"
	@echo "  make prepare_data     - Prepare the dataset"
	@echo "  make train_model      - Train the LSTM model"
	@echo "  make register_model   - Register the best model with MLflow"
	@echo "  make docker_up        - Build and run Docker containers"
	@echo "  make docker_down      - Stop and remove Docker containers"
	@echo "  make deploy_cloud     - Deploy the application to AWS using CloudFormation"
	@echo "  make delete_cloud     - Delete the AWS CloudFormation stack"
	@echo "  make help             - Display this help message"