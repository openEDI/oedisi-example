terraform {
  required_version = ">= 0.13.1"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 3.30"
    }
  }
}

provider "aws" {
  profile = "default"
  region = var.aws_region
  default_tags {
    tags = var.billing_tags
  }
}

variable "aws_region" {
  type = string
  default = "us-west-2"
}

variable "security_groups" {
  type = list(string)
}

variable "subnets" {
  type = list(string)
}

variable "billing_tags" {
  description = "Tags to set for all resources"
  type        = map(string)
}

resource "aws_ecr_repository" "nrel_gadal" {
  name                 = "nrel-gadal-ecr"
  image_tag_mutability = "MUTABLE"
}

output "ecr_repo_url" {
  value = aws_ecr_repository.nrel_gadal.repository_url
}
