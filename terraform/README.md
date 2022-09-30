# Running the docker container in the cloud

## Software Requirements

- terraform
- awscli
- gadal-example:0.0.0 as a build script

## Process

1. Create a `terraform.tfvars` command in this directory with `subnet_ids`, `security_groups`, and `billing_tags`.

2. `setup_run.sh` runs terraform and the necessary `aws` and `docker` commands to setup ECR, AWS Batch,
push to the registry, and submit a job.

3. `tear_down.sh` runs the relevant `terraform destroy` commands. You may need to delete ECR manually if it pushed correctly.
