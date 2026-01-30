# Overview

This document describes how to release packages from this repository.

## Releasing GenAI Packages

How to release a new version of the GenAI packages:

### Package Information

| Package Name | Path | Current Version | Tag Format                          |
|-------------|------|-----------------|-------------------------------------|
| splunk-otel-util-genai | `util/opentelemetry-util-genai` | 0.1.9           | `util-genai-v0.Y.Z`                 |
| splunk-otel-util-genai-evals | `util/opentelemetry-util-genai-evals` | 0.1.7           | `util-genai-evals-v0.Y.Z`           |
| splunk-otel-genai-emitters-splunk | `util/opentelemetry-util-genai-emitters-splunk` | 0.1.6           | `genai-emitters-splunk-v0.Y.Z`      |
| splunk-otel-genai-evals-deepeval | `util/opentelemetry-util-genai-evals-deepeval` | 0.1.12          | `genai-evals-deepeval-v0.Y.Z`       |
| splunk-otel-util-genai-translator-traceloop | `util/opentelemetry-util-genai-traceloop-translator` | 0.1.7           | `genai-translator-traceloop-v0.Y.Z` |
| splunk-otel-util-genai-translator-langsmith | `util/opentelemetry-util-genai-langsmith-translator` | 0.1.0           | `genai-translator-langsmith-v0.Y.Z` |
| splunk-otel-util-genai-translator-openlit | `util/opentelemetry-util-genai-openlit-translator` | 0.1.1           | `genai-translator-openlit-v0.Y.Z` |
| splunk-otel-instrumentation-langchain | `instrumentation-genai/opentelemetry-instrumentation-langchain` | 0.1.7           | `instrumentation-langchain-v0.Y.Z`  |
| splunk-otel-instrumentation-llamaindex | `instrumentation-genai/opentelemetry-instrumentation-llamaindex` | 0.1.0           | `instrumentation-llamaindex-v0.Y.Z`  |
| splunk-otel-instrumentation-aidefense | `instrumentation-genai/opentelemetry-instrumentation-aidefense` | 0.2.0           | `instrumentation-aidefense-v0.Y.Z`  |
| splunk-otel-instrumentation-weaviate | `instrumentation-genai/opentelemetry-instrumentation-weaviate` | 0.1.0           | `instrumentation-weaviate-v0.Y.Z`  |
| splunk-otel-instrumentation-crewai | `instrumentation-genai/opentelemetry-instrumentation-crewai` | 0.1.2           | `instrumentation-crewai-v0.Y.Z`  |
| splunk-otel-instrumentation-openai-agents | `instrumentation-genai/opentelemetry-instrumentation-openai-agents-v2` | 0.1.2           | `instrumentation-openai-agents-v0.Y.Z`  |
| splunk-otel-instrumentation-fastmcp | `instrumentation-genai/opentelemetry-instrumentation-fastmcp` | 0.1.1           | `instrumentation-fastmcp-v0.Y.Z`  |

### Release Steps

1) Create a new branch from `main`

2) Navigate to the package directory you want to release

3) Bump dependency versions in the package's `pyproject.toml`
    - update otel dependencies to the latest versions if needed

4) Bump the version in the package's `version.py` file
    - e.g., `util/opentelemetry-util-genai/src/opentelemetry/util/genai/version.py`
    - update `__version__ = "0.1.0"` to the new version

5) Add a new entry in CHANGELOG.md for the package release

6) Commit the changes with a message like "Bump util-genai version to 0.1.0"

7) Push the changes to the Github Splunk OTel Python repo

8) Open a PR and merge after approval

9) Navigate to the GitLab mirror and verify that the mirror has pulled the version you just merged by checking the
   version number in the package's `version.py` file

10) When ready to release, create a new tag with the **package-specific format** on main in GitLab
    - **IMPORTANT**: GenAI packages use a tag format that includes the package name prefix
    - Examples:
        - For `splunk-otel-util-genai` version `0.1.0`, create tag: `util-genai-v0.1.0`
        - For `splunk-otel-instrumentation-langchain` version `0.2.0`, create tag: `instrumentation-langchain-v0.2.0`
    - A tag matching the package-specific format will trigger the CI pipeline to build and publish that specific package to PyPI

11) Monitor the release pipeline in GitLab to ensure it completes successfully
    - The pipeline will show jobs like `build-util-genai`, `sign-util-genai`, `deploy-util-genai`

12) Post release, verify that the new package is available on PyPI
    - e.g., `https://pypi.org/project/splunk-otel-util-genai/0.1.0/`

13) Smoke test the release locally by installing the new package and running it with a small app
    ```bash
    pip install splunk-otel-util-genai==0.1.0
    ```

### Additional Steps for GA Releases Only

**Note**: Steps 14 and onward are only required for GA (Generally Available) releases. For alpha/beta releases, stop at step 13.

14) Navigate to Pipelines in the GitLab repo, click the download button for the build job that just ran,
    and select the package-specific build job artifact (e.g., `build-util-genai`)
    - this will download a tarball of the package files

15) Navigate to the Splunk OTel Python repo and create a New Release
    - create a new tag on publish with the tag name you created in step 10 (e.g., `util-genai-v0.1.0`)
    - set the title to that tag name
    - unpack the tarball from step 14 and drag its contents onto the attachments section of the New Release page
    - Leave the defaults selected and click Publish
