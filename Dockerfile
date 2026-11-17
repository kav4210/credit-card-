# AWS Lambda container image. Python 3.13 is required: the model was pickled with
# scikit-learn 1.9.1, which has no wheel for 3.12, and pickles must load under the same version.
# Build context needs artifacts/model.joblib and artifacts/threshold.json (run train.py first;
# they are gitignored).
FROM public.ecr.aws/lambda/python:3.13
COPY requirements-lambda.txt .
RUN pip install --no-cache-dir -r requirements-lambda.txt
COPY handler.py ${LAMBDA_TASK_ROOT}/
COPY artifacts/model.joblib artifacts/threshold.json ${LAMBDA_TASK_ROOT}/
CMD ["handler.lambda_handler"]
