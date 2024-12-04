import boto3
from marketing.constants import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, REGION_NAME



class S3Client:
    s3_client = None
    s3_resourse = None
    def __init__(self, region_name=REGION_NAME):
        if AWS_ACCESS_KEY_ID is None:
            raise Exception("AWS_ACCESS_KEY_ID environment variable not set")
        if AWS_SECRET_ACCESS_KEY is None:
            raise Exception("AWS_SECRET_ACCESS_KEY environment variable not set")
        try:
            if S3Client.s3_client is None or S3Client.s3_resourse is None:
                S3Client.s3_client = boto3.client(
                    's3',
                    aws_access_key_id=AWS_ACCESS_KEY_ID,
                    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                    region_name=region_name
                    )
            S3Client.s3_resourse = boto3.resource(
                's3',
                aws_access_key_id=AWS_ACCESS_KEY_ID,
                aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                region_name=region_name
                )
            self.s3_resourse = S3Client.resourse
            self.s3_client = S3Client.s3_client
        except Exception as e:
            raise e