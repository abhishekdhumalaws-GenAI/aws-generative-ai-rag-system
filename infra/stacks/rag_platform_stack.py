from aws_cdk import Stack, CfnOutput, aws_s3 as s3, aws_s3_notifications as s3n
from constructs import Construct

from stacks.storage_stack import StorageConstruct
from stacks.security_stack import SecurityConstruct
from stacks.search_stack import SearchConstruct
from stacks.compute_stack import ComputeConstruct
from stacks.api_stack import ApiConstruct
from stacks.monitoring_stack import MonitoringConstruct


class RagPlatformStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs):
        super().__init__(scope, construct_id, **kwargs)

        storage = StorageConstruct(self, "Storage")
        security = SecurityConstruct(self, "Security")
        search = SearchConstruct(
            self,
            "Search",
            lambda_role=security.lambda_role,
        )
        compute = ComputeConstruct(
            self,
            "Compute",
            lambda_role=security.lambda_role,
            opensearch_host=search.domain_endpoint,
            guardrail_id=security.guardrail_id,
            guardrail_version=security.guardrail_version_number,
        )

        storage.documents_bucket.grant_read_write(compute.document_processor)
        storage.documents_bucket.add_event_notification(
            s3.EventType.OBJECT_CREATED,
            s3n.LambdaDestination(compute.document_processor),
        )
        storage.chat_history_table.grant_read_write_data(compute.answer_api)

        api = ApiConstruct(self, "Api", answer_lambda=compute.answer_api)

        MonitoringConstruct(
            self,
            "Monitoring",
            document_processor=compute.document_processor,
            query_api=compute.query_api,
            answer_api=compute.answer_api,
            api=api.api,
        )

        CfnOutput(self, "DocumentsBucketName", value=storage.documents_bucket.bucket_name)
        CfnOutput(self, "OpenSearchEndpoint", value=search.domain_endpoint)
        CfnOutput(self, "ApiUrl", value=api.api.url)
