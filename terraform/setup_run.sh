#!/bin/bash

cd ecr_setup
terraform apply -var-file="../terraform.tfvars"
REPO_URL="$(terraform output -raw ecr_repo_url)"
echo "REPO_URL=$REPO_URL"

docker tag gadal-example:0.0.0 "$REPO_URL"
echo "Tagged"
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin $REPO_URL || { echo 'failed to login to ecr' ; exit 1; }
docker push $REPO_URL


cd ../batch_setup
terraform apply -var-file="../terraform.tfvars"

aws batch submit-job --job-name nrel-gadal-batch --job-queue oedi-gadal-queue --job-definition nrel-gadal-job
