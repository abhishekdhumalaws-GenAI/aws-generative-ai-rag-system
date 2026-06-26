from aws_cdk import (
    Duration,
    RemovalPolicy,
    CustomResource,
    aws_opensearchservice as opensearch,
    aws_lambda as _lambda,
)
from constructs import Construct


class SearchConstruct(Construct):
    def __init__(self, scope: Construct, construct_id: str, lambda_role):
        super().__init__(scope, construct_id)

        self.domain = opensearch.Domain(
            self,
            "RagOpenSearchDomain",
            version=opensearch.EngineVersion.OPENSEARCH_2_19,
            capacity=opensearch.CapacityConfig(
                data_nodes=1,
                data_node_instance_type="t3.small.search",
            ),
            ebs=opensearch.EbsOptions(
                volume_size=20,
            ),
            enforce_https=True,
            node_to_node_encryption=True,
            encryption_at_rest=opensearch.EncryptionAtRestOptions(enabled=True),
            removal_policy=RemovalPolicy.DESTROY,
        )

        self.domain_endpoint = self.domain.domain_endpoint

        dependency_layer = _lambda.LayerVersion(
            self,
            "IndexCreatorDependencyLayer",
            code=_lambda.Code.from_asset("../layers"),
            compatible_runtimes=[_lambda.Runtime.PYTHON_3_12],
            description="Dependencies for OpenSearch index creator",
        )

        index_creator = _lambda.Function(
            self,
            "OpenSearchIndexCreatorLambda",
            function_name="opensearch-index-creator-cdk",
            runtime=_lambda.Runtime.PYTHON_3_12,
            handler="lambda_function.lambda_handler",
            code=_lambda.Code.from_asset("../backend/opensearch-index-creator"),
            role=lambda_role,
            timeout=Duration.minutes(5),
            memory_size=256,
            layers=[dependency_layer],
        )

        self.index_resource = CustomResource(
            self,
            "RagVectorIndex",
            service_token=index_creator.function_arn,
            properties={
                "OpenSearchHost": self.domain_endpoint,
                "IndexName": "rag-vector-index",
            },
        )

        self.index_resource.node.add_dependency(self.domain)
