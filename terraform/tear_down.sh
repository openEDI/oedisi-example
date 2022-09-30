#!/bin/bash

cd batch_setup
terraform destroy -var-file="../terraform.tfvars"

cd ../ecr_setup
terraform destroy -var-file="../terraform.tfvars"

