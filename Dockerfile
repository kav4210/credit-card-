# Container image for AWS Lambda. scikit-learn is too large for a plain zip, so use an image.
FROM public.ecr.aws/lambda/python:3.12
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY handler.py ${LAMBDA_TASK_ROOT}/
COPY artifacts/model.joblib artifacts/threshold.json ${LAMBDA_TASK_ROOT}/
CMD ["handler.lambda_handler"]
