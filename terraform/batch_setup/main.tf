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

resource "aws_iam_role" "ecs_instance_role" {
  name = "ecs_instance_role"

  assume_role_policy = <<EOF
{
    "Version": "2012-10-17",
    "Statement": [
    {
        "Action": "sts:AssumeRole",
        "Effect": "Allow",
        "Principal": {
            "Service": "ec2.amazonaws.com"
        }
    }
    ]
}
EOF
}

resource "aws_iam_role_policy_attachment" "ecs_instance_role" {
  role       = aws_iam_role.ecs_instance_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonEC2ContainerServiceforEC2Role"
}

resource "aws_iam_instance_profile" "ecs_instance_role" {
  name = "ecs_instance_role"
  role = aws_iam_role.ecs_instance_role.name
}

resource "aws_iam_role" "aws_batch_service_role" {
  name = "aws_batch_service_role"

  assume_role_policy = <<EOF
{
    "Version": "2012-10-17",
    "Statement": [
    {
        "Action": "sts:AssumeRole",
        "Effect": "Allow",
        "Principal": {
        "Service": "batch.amazonaws.com"
        }
    }
    ]
}
EOF
}

resource "aws_iam_role_policy_attachment" "aws_batch_service_role" {
  role       = aws_iam_role.aws_batch_service_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSBatchServiceRole"
}

resource "aws_batch_compute_environment" "gadal" {
  compute_environment_name = "gadal-spot-fleet"

  compute_resources {
    allocation_strategy = "SPOT_CAPACITY_OPTIMIZED"
    instance_role = aws_iam_instance_profile.ecs_instance_role.arn

    instance_type = [
      "optimal",
    ]

    max_vcpus = 4
    min_vcpus = 0

    security_group_ids = var.security_groups
    subnets = var.subnets

    type = "SPOT"
  }

  service_role = aws_iam_role.aws_batch_service_role.arn
  type         = "MANAGED"
  depends_on   = [aws_iam_role_policy_attachment.aws_batch_service_role]
}

resource "aws_batch_job_queue" "gadal" {
  name     = "oedi-gadal-queue"
  state    = "ENABLED"
  priority = 1
  compute_environments = [
    "${aws_batch_compute_environment.gadal.arn}"
  ]
}

data "aws_ecr_repository" "nrel_gadal" {
  name = "nrel-gadal-ecr"
}

resource "aws_batch_job_definition" "nrel_gadal_job" {
  name = "nrel-gadal-job"
  type = "container"
  parameters = {}
  container_properties = <<CONTAINER_PROPERTIES
{
  "image": "${data.aws_ecr_repository.nrel_gadal.repository_url}:latest",
  "jobRoleArn": "${aws_iam_role.job_role.arn}",
  "vcpus": 1,
  "memory": 1024,
  "environment": [],
  "volumes": [],
  "mountPoints": [],
  "command": []
}
CONTAINER_PROPERTIES
}



# IAM Role for jobs
resource "aws_iam_role" "job_role" {
  name               = "job_role"
  assume_role_policy = <<EOF
{
    "Version": "2012-10-17",
    "Statement":
    [
      {
          "Action": "sts:AssumeRole",
          "Effect": "Allow",
          "Principal": {
            "Service": "ecs-tasks.amazonaws.com"
          }
      }
    ]
}
EOF
}


# # S3 read/write policy
# resource "aws_iam_policy" "s3_policy" {
#   name   = "s3_policy"
#   policy = <<EOF
# {
#   "Version": "2012-10-17",
#   "Statement": [
#     {
#         "Effect": "Allow",
#         "Action": [
#             "s3:Get*",
#             "s3:List*",
#             "s3:Put*"
#         ],
#         "Resource": [
#           "${aws_s3_bucket.results_s3.arn}",
#           "${aws_s3_bucket.results_s3.arn}/*"
#         ]
#     }
#   ]
# }
# EOF
# }

# # Attach the policy to the job role
# resource "aws_iam_role_policy_attachment" "job_policy_attachment" {
#   role       = aws_iam_role.job_role.name
#   policy_arn = aws_iam_policy.s3_policy.arn
# }
