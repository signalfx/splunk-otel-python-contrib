#!/usr/bin/env bash
set -e

cd docker

# Parse arguments
release_tag="$1"        # e.g. util-genai-v1.2.3 or instrumentation-langchain-v1.2.3
package_name="$2"       # e.g. util-genai, instrumentation-langchain
pypi_package_name="$3"  # e.g. splunk-otel-util-genai, splunk-otel-instrumentation-langchain

# Extract version from tag (remove package prefix)
# e.g. util-genai-v1.2.3 -> 1.2.3
version=$(echo $release_tag | sed "s/^${package_name}-v//")
major_version=$(echo $version | cut -d '.' -f1) # e.g. "1"

# Determine repository name based on package
repo="quay.io/signalfx/splunk-otel-${package_name}"

echo ">>> Publishing Docker image for package: $package_name"
echo ">>> Release tag: $release_tag"
echo ">>> Version: $version"
echo ">>> PyPI package name: $pypi_package_name"
echo ">>> Docker repository: $repo"

check_package_available() {
  max_attempts=10
  sleep_seconds=10

  echo "Waiting for $pypi_package_name==$version to be available on PyPI..."

  for i in $(seq 1 $max_attempts); do
      if curl --silent --fail "https://pypi.org/pypi/$pypi_package_name/$version/json" > /dev/null; then
          echo "Package $pypi_package_name==$version is available on PyPI."
          break
      fi
      echo "Attempt $i: Package not yet available. Retrying in $sleep_seconds seconds..."
      sleep $sleep_seconds
  done

  if [ "$i" -eq "$max_attempts" ]; then
      echo "ERROR: Package $pypi_package_name==$version was not found on PyPI after $max_attempts attempts."
      exit 1
  fi
}

build_docker_image() {
  echo ">>> Building the operator docker image for $package_name..."
  
  # Update requirements.txt with the specific package version
  echo "$pypi_package_name==$version" > requirements.txt
  
  docker build -t splunk-otel-${package_name} .
  docker tag splunk-otel-${package_name} ${repo}:latest
  docker tag splunk-otel-${package_name} ${repo}:${major_version}
  docker tag splunk-otel-${package_name} ${repo}:${version}
  docker tag splunk-otel-${package_name} ${repo}:${release_tag}
}

login_to_quay_io() {
  echo ">>> Logging into quay.io ..."
  docker login -u "$QUAY_USERNAME" -p "$QUAY_PASSWORD" quay.io
}

publish_docker_image() {
  echo ">>> Publishing the operator docker image for $package_name..."
  docker push ${repo}:latest
  docker push ${repo}:${major_version}
  docker push ${repo}:${version}
  docker push ${repo}:${release_tag}
}

check_package_available
build_docker_image
login_to_quay_io
publish_docker_image
