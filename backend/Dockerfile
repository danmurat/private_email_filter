#
# Dockerfile for the server ONLY
# this container will be what runs on AWS
# multi-stage build. First build to check server logic tests. If they pass, we build
# the final 'prod' version (which excludes tests and keeps image lean)
#
FROM python:3.11-slim AS tester

WORKDIR /app
COPY test_requirements.txt .
RUN pip install --no-cache-dir -r test_requirements.txt
COPY . .
RUN python -m pytest --ignore=test/network/ && touch /tmp/tests_passed
# docker is weird and skips building tester if prod has no dependency with it.
# so if we create a file once passed, and copy that from prod, it will make
# sure docker first runs tester.


# if tests pass we build the final, lean image
FROM python:3.11-slim AS prod

# avoids bloat (with pyc files) and helps with error messages in docker
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app 
# will make sure tester runs first
COPY --from=tester ./tmp/tests_passed ./tmp/tests_passed
COPY prod_requirements.txt .
RUN pip install --no-cache-dir -r prod_requirements.txt
COPY ./src/ ./src/
COPY ./models/ ./models/
# Expose the port for documentation/AWS mapping
EXPOSE ${PORT}
# allows us to access the localhost outside the internal network (needed for AWS)
# TODO: look into the security of this
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
